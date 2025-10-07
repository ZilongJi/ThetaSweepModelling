from dataclasses import dataclass
import brainpy as bp
import brainpy.math as bm
import jax
import numpy as np
import networkx as nx

@dataclass
class PCParams:
    tau: float = 10.0
    tau_v: float = 100.0
    noise_strength: float = 0.0
    k: float = 0.2
    adaptation_strength: float = 15.0
    a: float = 0.2
    A: float = 5.0
    J0: float = 1.0
    g: float = 1.0
    conn_noise: float = 0.0

class PCNet(bp.DynamicalSystem):
    """
    Graph-based continuous-attractor place cell network.
    Each node in G corresponds to a place cell.
    Connectivity is a Gaussian function of geodesic distance on the graph.
    """
    def __init__(self, Graph, params: PCParams = PCParams()):
        super().__init__()
        self.Graph = Graph
        self.params = params
        
        # number of cells = number of nodes in graph
        self.cell_num = len(Graph.nodes)
        self.node_list = list(Graph.nodes)
        
        dx = self.Graph.graph["dx"] 
        self.x = bm.asarray(np.arange(self.cell_num) * dx)
        
        # --- derived parameters ---
        self.m = params.adaptation_strength * params.tau / params.tau_v

        # --- compute geodesic distance matrix (cell_num×cell_num)
        geodist = dict(nx.all_pairs_dijkstra_path_length(Graph, weight='weight'))
        D = np.zeros((self.cell_num, self.cell_num))    
        for i, ni in enumerate(self.node_list):
            for j, nj in enumerate(self.node_list):
                D[i, j] = geodist[ni][nj]
        self.D = bm.asarray(D)        
        
        # --- build connectivity based on geodesic distance ---
        base_connection = self.make_connection(self.D)
        noise_connection = np.random.normal(0, params.conn_noise, size=(self.cell_num, self.cell_num))
        self.conn_mat = base_connection + noise_connection

        # --- state variables ---
        self.r = bm.Variable(bm.zeros(self.cell_num))
        self.u = bm.Variable(bm.zeros(self.cell_num))
        self.v = bm.Variable(bm.zeros(self.cell_num))
        self.center = bm.Variable(bm.zeros(1))

        # --- integrator ---
        self.integral = bp.odeint(method="exp_euler", f=self.derivative)

    @property
    def derivative(self):
        du = lambda u, t, input: (-u + input - self.v) / self.params.tau
        dv = lambda v, t: (-v + self.m * self.u) / self.params.tau_v
        return bp.JointEq([du, dv])

    # ===== connectivity based on geodesic distances =====
    def make_connection(self, D):
        """
        Gaussian weight kernel based on graph geodesic distances.
        """
        a = self.params.a
        J0 = self.params.J0
        W = J0 * bm.exp(-0.5 * (D / a) ** 2)
        # normalise or scale if desired
        W = W / (bm.sqrt(2 * bm.pi) * a)
        return W

    def get_bump_center(self, r, x):
        denom = bm.sum(r) + 1e-12
        center = bm.sum(r * x) / denom
        return center.reshape(-1,)
        
    # ===== external input based on animal position =====
    def input_bump(self, animal_pos_node_index):
        """
        Generate Gaussian bump centred on the node corresponding to the animal's current position.
        node_index: integer index into self.nodes
        """
        d = self.D[animal_pos_node_index]
        return self.params.A * bm.exp(-0.5 * (d / self.params.a) ** 2)        

    # ===== update loop =====
    def update(self, animal_pos_node_index, ThetaInput):
        self.center.value = self.get_bump_center(r=self.r, x=self.x)
        Iext = ThetaInput * self.input_bump(animal_pos_node_index)
        Irec = bm.matmul(self.conn_mat, self.r)
        noise = bm.random.randn(self.cell_num) * self.params.noise_strength
        input_total = Iext + Irec + noise

        # integrate for current step
        u, v = self.integral(self.u, self.v, bp.share.load("t"), input_total)
        self.u.value = bm.where(u > 0, u, 0)
        self.v.value = v

        u_sq = bm.square(self.u)
        self.r.value = self.params.g * u_sq / (1.0 + self.params.k * bm.sum(u_sq))


@dataclass
class DCParams:
    tau: float = 10.0
    tau_v: float = 100.0
    noise_strength: float = 0.1
    k: float = 0.2
    adaptation_strength: float = 15.0
    a: float = 0.7
    A: float = 3.0
    J0: float = 1.0
    g: float = 1.0
    z_min: float = -bm.pi
    z_max: float = bm.pi
    conn_noise: float = 0.0


class DCNet(bp.DynamicalSystem):
    """
    1D continuous-attractor direction cell network
    """
    def __init__(self, cell_num: int, params: DCParams = DCParams()):
        super().__init__()
        self.cell_num = cell_num
        self.params = params

        # --- derived parameters ---
        self.m = params.adaptation_strength * params.tau / params.tau_v

        # --- feature space ---
        self.z_min = params.z_min
        self.z_max = params.z_max
        self.z_range = self.z_max - self.z_min
        x1 = bm.linspace(self.z_min, self.z_max, self.cell_num + 1)
        self.x = x1[:-1]

        # --- connectivity ---
        base_connection = self.make_connection()
        noise_connection = np.random.normal(0, params.conn_noise, size=(self.cell_num, self.cell_num))
        self.conn_mat = base_connection + noise_connection

        # --- state variables ---
        self.r = bm.Variable(bm.zeros(self.cell_num))
        self.u = bm.Variable(bm.zeros(self.cell_num))
        self.v = bm.Variable(bm.zeros(self.cell_num))
        self.center = bm.Variable(bm.zeros(1))

        # --- integrator ---
        self.integral = bp.odeint(method="exp_euler", f=self.derivative)

    @property
    def derivative(self):
        du = lambda u, t, input: (-u + input - self.v) / self.params.tau
        dv = lambda v, t: (-v + self.m * self.u) / self.params.tau_v
        return bp.JointEq([du, dv])

    # ===== utilities =====
    def handle_periodic_condition(self, A):
        B = bm.where(A > bm.pi, A - 2 * bm.pi, A)
        B = bm.where(B < -bm.pi, B + 2 * bm.pi, B)
        return B

    def calculate_dist(self, d):
        d = self.handle_periodic_condition(d)
        d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    # ===== connectivity =====
    def make_connection(self):
        @jax.vmap
        def get_J(xbins):
            d = self.calculate_dist(xbins - self.x)
            Jxx = (
                self.params.J0
                * bm.exp(-0.5 * bm.square(d / self.params.a))
                / (bm.sqrt(2 * bm.pi) * self.params.a)
            )
            return Jxx
        return get_J(self.x)

    # ===== dynamics =====
    def get_bump_center(self, r, x):
        exppos = bm.exp(1j * x)
        center = bm.angle(bm.sum(exppos * r))
        return center.reshape(-1,)

    def input_bump(self, head_direction):
        return self.params.A * bm.exp(
            -0.5 * bm.square(self.calculate_dist(self.x - head_direction) / self.params.a)
        )

    # ===== update loop =====
    def update(self, head_direction, ThetaInput):
        self.center.value = self.get_bump_center(r=self.r, x=self.x)
        Iext = ThetaInput * self.input_bump(head_direction)
        Irec = bm.matmul(self.conn_mat, self.r)
        noise = bm.random.randn(self.cell_num) * self.params.noise_strength
        input_total = Iext + Irec + noise

        # integrate for current step
        u, v = self.integral(self.u, self.v, bp.share.load("t"), input_total)
        self.u.value = bm.where(u > 0, u, 0)
        self.v.value = v

        u_sq = bm.square(self.u)
        self.r.value = self.params.g * u_sq / (1.0 + self.params.k * bm.sum(u_sq))

@dataclass
class GCParams:
    # dynamics
    tau: float = 10.0
    tau_v: float = 100.0
    noise_strength: float = 0.1 #activity noise
    conn_noise: float = 0.0 #connectivity noise
    k: float = 1.0
    adaptation_strength: float = 15.0  # (mbar)
    
    # connectivity / input 
    a: float = 0.8
    A: float = 3.0
    J0: float = 5.0
    g: float = 1000.0 #scale the firing rate to make it reasonable, no biological meaning

    #controlling grid spacing, larger means smaller spacing
    mapping_ratio: float = 1  
    
    #cntrolling offset length from conjunctive gc layer to gc layer, this is the key to drive the bump to move
    phase_offset: float = 1.0 / 20  #relative to -pi~pi range
    
class GCNet(bp.DynamicalSystem):
    """
    2D continuous-attractor grid cell network
    Grid size is num_gc_x x num_gc_x (total cells = num_gc_1side**2).
    """

    def __init__(self, num_dc: int = 100, num_gc_x: int = 100, envsize=1, params: GCParams = GCParams()):
        super().__init__()
        
        self.num_dc = num_dc       
        self.num_gc_1side = num_gc_x
        self.envsize = envsize
        self.params = params

        # ----- derived parameters -----
        self.num = num_gc_x * num_gc_x
        self.m = params.adaptation_strength * params.tau / params.tau_v
        self.Lambda = 2 * bm.pi / params.mapping_ratio #grid spacing

        # ----- coordinate transforms (hex vs rect) -----
        # Note that coor_transform is to map a parallelogram with a 60-degree angle back to a square
        # The logic is to partition the 2D space into parallelograms, each of which contains one lattice of grid cells, and repeat the parallelogram to tile the whole space
        self.coor_transform = bm.array([[1.0, -1.0 / bm.sqrt(3.0)],
                                        [0.0,  2.0 / bm.sqrt(3.0)]])
          
        # inverse, which is bm.array([[1.0, 1.0 / 2],[0.0,  bm.sqrt(3.0) / 2]])   
        # Note that coor_transform_inv is to map a square to a parallelogram with a 60-degree angle
        self.coor_transform_inv = np.linalg.inv(np.array(self.coor_transform))

        # ----- feature space -----
        x_bins = bm.linspace(-bm.pi, bm.pi, num_gc_x + 1)
        x_grid, y_grid = bm.meshgrid(x_bins[:-1], x_bins[:-1])
        self.x_grid = x_grid.reshape(-1)
        self.y_grid = y_grid.reshape(-1)

        # positions in (x,y) space and transformed space
        self.value_grid = bm.stack([self.x_grid, self.y_grid], axis=1)    # (N, 2)
        self.value_bump = self.value_grid * 4
        # ----- candidate centers (for center snapping) -----
        self.candidate_centers = self.make_candidate_centers(self.Lambda)

        # ----- connectivity -----
        base_connection = self.make_connection()
        noise_connection = np.random.normal(0.0, params.conn_noise, size=(self.num, self.num))
        self.conn_mat = base_connection + noise_connection

        # ----- state variables -----
        self.r = bm.Variable(bm.zeros(self.num))
        self.u = bm.Variable(bm.zeros(self.num))
        self.v = bm.Variable(bm.zeros(self.num))
        self.gc_bump = bm.Variable(bm.zeros(self.num))
        self.conj_input = bm.Variable(bm.zeros(self.num))
        self.center_phase = bm.Variable(bm.zeros(2))
        self.center_position = bm.Variable(bm.zeros(2))

        # ----- integrator -----
        self.integral = bp.odeint(method="exp_euler", f=self.derivative)

    # ========================= Dynamics =========================
    @property
    def derivative(self):
        # pass total input (Irec + external + noise) as 'inp'
        du = lambda u, t, inp: (-u + inp - self.v) / self.params.tau
        dv = lambda v, t: (-v + self.m * self.u) / self.params.tau_v
        return bp.JointEq([du, dv])

    # ========================= Utilities =========================
    def handle_periodic_condition(self, d):
        d = bm.where(d > bm.pi, d - 2.0 * bm.pi, d)
        d = bm.where(d < -bm.pi, d + 2.0 * bm.pi, d)
        return d

    def calculate_dist(self, d):
        """
        d: (..., 2) displacement in original (x,y).
        Return Euclidean distance after transform (hex/rect).
        """
        #consider the periodic boundary condition
        d = self.handle_periodic_condition(d)
        # transform to lattice axes
        dist = (bm.matmul(self.coor_transform_inv, d.T)).T  #This means the bump on the parallelogram lattice is a Gaussian, while in the square space it is a twisted Gaussian
        return bm.sqrt(dist[:, 0] ** 2 + dist[:, 1] ** 2)

    def make_candidate_centers(self, Lambda):
        #This will generate a massivly large number of candidate centers, so that cover the simulated environmental size sufficiently
        N_c = bm.int(bm.ceil(self.envsize/Lambda)+1) * 2  #number of candidates along one dimension
        cc = bm.zeros((N_c, N_c, 2))
 
        for i in range(N_c):
            for j in range(N_c):
                cc = cc.at[i, j, 0].set((-N_c // 2 + i) * Lambda)
                cc = cc.at[i, j, 1].set((-N_c // 2 + j) * Lambda)
        
        cc_tranformed = bm.dot(self.coor_transform_inv, cc.reshape(N_c * N_c, 2).T).T
        
        return cc_tranformed

    # ========================= Connectivity =========================
    def make_connection(self):
        @jax.vmap
        def kernel(v):
            # v: (2,) location in (x,y)
            d = self.calculate_dist(v - self.value_grid)  # (N,)
            return (
                (self.params.J0 / self.params.g)
                * bm.exp(-0.5 * bm.square(d / self.params.a))
                / (bm.sqrt(2.0 * bm.pi) * self.params.a)
            )
        return kernel(self.value_grid)  # (N, N)

    # ========================= Inputs =========================
    def position2phase(self, position):
        """
        map position->phase; phase is wrapped to [-pi, pi] per-axis
        """
        mapped_pos = position * self.params.mapping_ratio
        phase = bm.matmul(self.coor_transform, mapped_pos) + bm.pi
        px = bm.mod(phase[0], 2.0 * bm.pi) - bm.pi
        py = bm.mod(phase[1], 2.0 * bm.pi) - bm.pi
        return bm.array([px, py])

    def calculate_input_from_conjgc(self, animal_pos, direction_activity, theta_modulation):
        """Get input from conjunctive grid cell layer → grid cell layer; returns (N,) vector."""
        assert bm.size(animal_pos) == 2
        num_dc = self.num_dc
        num_gc = self.num
        direction_bin = bm.linspace(-bm.pi, bm.pi, num_dc)

        # # lag relative to head direction
        # lagvec = -bm.array([bm.cos(head_direction), bm.sin(head_direction)]) * self.params.phase_offset * 1.4
        # offset = bm.array([bm.cos(direction_bin), bm.sin(direction_bin)]) * self.params.phase_offset + lagvec.reshape(-1, 1)
        
        offset = bm.array([bm.cos(direction_bin), bm.sin(direction_bin)]) * self.params.phase_offset

        center_conj = self.position2phase(animal_pos.reshape(-1, 1) + offset.reshape(-1, num_dc))
        
        conj_input = bm.zeros((num_dc, num_gc))
        for i in range(num_dc):
            d = self.calculate_dist(bm.asarray(center_conj[:, i]) - self.value_grid)
            conj_input = conj_input.at[i].set(self.params.A * bm.exp(-0.5 * bm.square(d / self.params.a)))

        # weighting by direction bump activity: keep top one-third (by max) then normalize, I thinking using all direction_activity should also be fine
        weight = bm.where(direction_activity > bm.max(direction_activity) / 3.0, direction_activity, 0.0)
        weight = weight / (bm.sum(weight) + 1e-12) # avoid div-by-zero, dim: (num_dc,)

        return (bm.matmul(conj_input.T, weight).reshape(-1) * theta_modulation)  #dim: (num_gc, num_dc) x (num_dc,) -> (num_gc,)

    # ========================= Bump center (phase/pos) =========================
    def get_unique_activity_bump(self, network_activity, animal_position):
        """
        Estimate a unique bump (activity peak) from the current network state,
        given the animal's actual position.

        Returns
        -------
        center_phase : (2,) array
            Phase coordinates of bump center on the manifold.
        center_position : (2,) array
            Real-space position of the bump (nearest candidate).
        bump : (N,) array
            Gaussian bump template centered at center_position.
        """

        # --- find bump center in phase space ---
        exppos_x = bm.exp(1j * self.x_grid)
        exppos_y = bm.exp(1j * self.y_grid)
        activity_masked = bm.where(network_activity > bm.max(network_activity) * 0.1, network_activity, 0.0)

        center_phase = bm.zeros((2,))
        center_phase = center_phase.at[0].set(bm.angle(bm.sum(exppos_x * activity_masked)))
        center_phase = center_phase.at[1].set(bm.angle(bm.sum(exppos_y * activity_masked)))

        # --- map back to real space, snap to nearest candidate ---
        center_phase_residual = bm.matmul(self.coor_transform_inv, center_phase) / self.params.mapping_ratio
        candidate_pos_all = self.candidate_centers + center_phase_residual
        distances = bm.linalg.norm(candidate_pos_all - animal_position, axis=1)
        center_position = candidate_pos_all[bm.argmin(distances)]

        # --- build Gaussian bump template ---
        d = bm.asarray(center_position) - self.value_bump
        dist = bm.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        gc_bump = self.params.A * bm.exp(-bm.square(dist / self.params.a))

        return center_phase, center_position, gc_bump

    # one-step update (main)
    def update(self, animal_posistion, direction_activity, theta_modulation):
        
        # get bump activity in real space info from network activity on the manifold ---
        center_phase, center_position, gc_bump = self.get_unique_activity_bump(self.r, animal_posistion)
        self.center_phase.value = center_phase
        self.center_position.value = center_position
        self.gc_bump.value = gc_bump

        # get external input to grid cell layer from conjunctive grid cell layer
        # note that this conjunctive input will be theta modulated. When speed is high, theta modulation is high, thus input is stronger
        # This is how we get longer theta sweeps when speed is high
        conj_input = self.calculate_input_from_conjgc(animal_posistion, direction_activity, theta_modulation)
        self.conj_input.value = conj_input

        # recurrent + noise
        Irec = bm.matmul(self.conn_mat, self.r)
        input_noise = bm.random.randn(self.num) * self.params.noise_strength
        total_net_input = Irec + conj_input + input_noise

        # integrate
        u, v = self.integral(self.u, self.v, bp.share.load("t"), total_net_input)
        self.u.value = bm.where(u > 0.0, u, 0.0)
        self.v.value = v

        # get neuron firing by global inhibition
        u_sq = bm.square(self.u)
        self.r.value = self.params.g * u_sq / (1.0 + self.params.k * bm.sum(u_sq))
