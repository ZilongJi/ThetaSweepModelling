from dataclasses import dataclass
import brainpy as bp
import brainpy.math as bm
import jax
import numpy as np


@dataclass
class HDParams:
    tau: float = 10.0
    tau_v: float = 100.0
    noise_strength: float = 0.01
    k: float = 0.2
    adaptation_strength: float = 15.0
    a: float = 0.7
    A: float = 3.0
    J0: float = 1.0
    z_min: float = -bm.pi
    z_max: float = bm.pi
    conn_noise: float = 0.0


class HDNet(bp.DynamicalSystem):
    def __init__(self, cell_num: int, params: HDParams = HDParams()):
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
        self.rho = self.cell_num / self.z_range
        self.dx = self.z_range / self.cell_num

        # --- connectivity ---
        base_connection = self.make_connection()
        noise_connection = np.random.normal(0, params.conn_noise, size=(self.cell_num, self.cell_num))
        self.conn_mat = base_connection + noise_connection

        # --- state variables ---
        self.r = bm.Variable(bm.zeros(self.cell_num))
        self.u = bm.Variable(bm.zeros(self.cell_num))
        self.v = bm.Variable(bm.zeros(self.cell_num))
        self.center = bm.Variable(bm.zeros(1))
        self.centerI = bm.Variable(bm.zeros(1))

        # --- integrator ---
        self.integral = bp.odeint(method="exp_euler", f=self.derivative)

    @property
    def derivative(self):
        du = lambda u, t, input: (-u + input - self.v) / self.params.tau
        dv = lambda v, t: (-v + self.m * self.u) / self.params.tau_v
        return bp.JointEq([du, dv])

    # ===== utilities =====
    def periodic(self, A):
        B = bm.where(A > bm.pi, A - 2 * bm.pi, A)
        B = bm.where(B < -bm.pi, B + 2 * bm.pi, B)
        return B

    def calculate_dist(self, d):
        d = self.periodic(d)
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

    def input_bump(self, HD):
        return self.params.A * bm.exp(
            -0.5 * bm.square(self.calculate_dist(self.x - HD) / self.params.a)
        )

    # ===== state management =====
    def reset_state(self):
        self.r[:] = 0.0
        self.u[:] = 0.0
        self.v[:] = 0.0
        self.center[:] = 0.0

    # ===== update loop =====
    def update(self, direction, ThetaInput):
        self.center.value = self.get_bump_center(r=self.r, x=self.x)
        Iext = ThetaInput * self.input_bump(direction)
        Irec = bm.matmul(self.conn_mat, self.r)
        noise = bm.random.randn(self.cell_num) * self.params.noise_strength
        input_total = Iext + Irec + noise

        # integrate for current step
        u, v = self.integral(self.u, self.v, bp.share.load("t"), input_total)
        self.u.value = bm.where(u > 0, u, 0)
        self.v.value = v

        r1 = bm.square(self.u)
        self.r.value = r1 / (1.0 + self.params.k * bm.sum(r1))


class GD_cell(bp.DynamicalSystem):
    def __init__(
        self,
        ratio=1,
        noise_stre=0.01,
        num=100,
        tau=10.0,
        tau_v=100.0,
        mbar=75.0,
        a=0.5,
        A=1.0,
        J0=5.0,
        k=1,
        g = 1000,
        x_min=-bm.pi,
        x_max=bm.pi,
        num_hd = 100,
        Phase_Offset = 1/9,
        Grid = 'Rectangle',
        conn_noise = 0.,
    ):
        super(GD_cell, self).__init__()

        # dynamics parameters
        self.tau = tau  # The synaptic time constant
        self.tau_v = tau_v  # The time constant of the adaptation variable
        self.ratio = ratio
        self.num_x = num  # number of excitatory neurons for x dimension
        self.num_y = num  # number of excitatory neurons for y dimension
        self.num = self.num_x * self.num_y
        self.num_hd = num_hd
        self.k = k  # Degree of the rescaled inhibition
        self.a = a  # Half-width of the range of excitatory connections
        self.A = A  # Magnitude of the external input
        self.g = g
        self.J0 = J0/g  # maximum connection value
        self.m = mbar * tau / tau_v
        self.noise_stre = noise_stre
        self.Lambda = 2*bm.pi/self.ratio
        self.Phase_Offset = Phase_Offset

        # feature space
        self.x_range = x_max - x_min
        phi_x = bm.linspace(x_min, x_max, self.num_x + 1)  # The encoded feature values
        self.x = phi_x[0:-1]
        self.y_range = self.x_range
        phi_y = bm.linspace(x_min, x_max, self.num_y + 1)  # The encoded feature values
        self.y = phi_y[0:-1]
        x_grid, y_grid = bm.meshgrid(self.x, self.y)
        self.x_grid = x_grid.flatten()
        self.y_grid = y_grid.flatten()
        self.value_grid = bm.stack([self.x_grid, self.y_grid]).T
        self.value_bump = self.value_grid * 4
        self.rho = self.num / (self.x_range * self.y_range)  # The neural density
        self.dxy = 1 / self.rho  # The stimulus density

        N_c = 32
        Candidate_center = bm.zeros([N_c,N_c,2])

        if Grid == 'Hexagonal':
            self.coor_transform = bm.array([[1 , -1/bm.sqrt(3)],[0, 2/bm.sqrt(3)]])
            self.coor_transform_inv = np.linalg.inv(self.coor_transform)
            for ni in range(N_c):
                for nj in range(N_c):
                    Candidate_center[ni,nj,0] = (-int(N_c/2)+ni)*self.Lambda + (-2+nj)*self.Lambda/2
                    Candidate_center[ni,nj,1] = (-int(N_c/2)+nj)*self.Lambda * bm.sqrt(3)/2
            self.Candidate_center = Candidate_center.reshape([N_c**2,2])

        if Grid == 'Rectangle':
            self.coor_transform = bm.array([[1, 0], [0, 1]])
            self.coor_transform_inv = self.coor_transform
            for ni in range(N_c):
                for nj in range(N_c):
                    Candidate_center[ni,nj,0] = (-int(N_c/2)+ni)*self.Lambda
                    Candidate_center[ni,nj,1] = (-int(N_c/2)+nj)*self.Lambda
            self.Candidate_center = Candidate_center.reshape([N_c**2,2])
        self.pos_grid = bm.matmul(self.coor_transform_inv, bm.transpose(self.value_grid)).T
        

        # initialize conn matrix
        conn_mat = self.make_conn()
        # 生成一个100x100的随机高斯矩阵，均值为0，标准差为1
        gaussian_matrix = np.random.normal(loc=0, scale=conn_noise, size=(self.num, self.num))
        self.conn_mat = gaussian_matrix + conn_mat
        # initialize dynamical variables
        self.r = bm.Variable(bm.zeros(self.num))
        self.u = bm.Variable(bm.zeros(self.num))
        self.v = bm.Variable(bm.zeros(self.num))
        self.bump = bm.Variable(bm.zeros(self.num))
        self.input = bm.Variable(bm.zeros(self.num))
        self.center_phase = bm.Variable(bm.zeros(2))
        self.center_pos = bm.Variable(bm.zeros(2))
        self.center_I = bm.Variable(bm.zeros(2))

        # 定义积分器
        self.integral = bp.odeint(method="exp_euler", f=self.derivative)

    def circle_period(self, d):
        d = bm.where(d > bm.pi, d - 2 * bm.pi, d)
        d = bm.where(d < -bm.pi, d + 2 * bm.pi, d)
        return d

    def dist(self, d):
        d = self.circle_period(d)
        dis = bm.matmul(self.coor_transform_inv, bm.transpose(d)).T
        delta_x = dis[:, 0]
        delta_y = dis[:, 1]
        dis = bm.sqrt(delta_x ** 2 + delta_y ** 2)
        return dis

    def make_conn(self):
        @jax.vmap
        def get_J(v):
            d = self.dist(v - self.value_grid)
            Jxx = (
                self.J0
                * bm.exp(-0.5 * bm.square(d / self.a))
                / (bm.sqrt(2 * bm.pi) * self.a)
            )
            return Jxx

        return get_J(self.value_grid)

    def Postophase(self, pos):
        Loc = pos * self.ratio# ratio = inverse of grid scale
        phase = bm.matmul(self.coor_transform, Loc) + bm.pi  # 坐标变换
        phase_x = bm.mod(phase[0], 2 * bm.pi) - bm.pi
        phase_y = bm.mod(phase[1], 2 * bm.pi) - bm.pi
        Phase = bm.array([phase_x, phase_y])
        return Phase
    

    def input_by_conjG_new(self, Animal_location, HD_activity, ThetaModulator, HD_truth):
        assert bm.size(Animal_location) == 2
        num_hd = self.num_hd
        hd = bm.linspace(-bm.pi,bm.pi,num_hd) 
        # each head-direction cell corresponds to a group of Conjunctive grid cells, which in turn projects to pure grid cells with assymetric connections determined by offset(hd)
        lagvec = -bm.array([bm.cos(HD_truth), bm.sin(HD_truth)]) * self.Phase_Offset * 1.4
        offset = bm.array([bm.cos(hd), bm.sin(hd)]) * self.Phase_Offset + lagvec.reshape(-1,1)
        self.center_conjG = self.Postophase(
            Animal_location.reshape([-1,1]) + offset.reshape(-1,num_hd)
        )  # Ideal phase using mapping function
        input = bm.zeros([num_hd, self.num])
        for i in range(num_hd):
            d = self.dist(bm.asarray(self.center_conjG[:,i]) - self.value_grid)
            input[i] = self.A * bm.exp(-0.25 * bm.square(d / self.a))
            
        max_hd = bm.max(HD_activity)
        hd_weight = bm.where(HD_activity>max_hd/3, HD_activity, 0)
        hd_weight = hd_weight/bm.sum(hd_weight)

        total_input = bm.matmul(input.transpose(), hd_weight).reshape(-1,) * ThetaModulator
        return total_input


    def get_center(self, pos):
        exppos_x = bm.exp(1j * self.x_grid)
        exppos_y = bm.exp(1j * self.y_grid)
        r = bm.where(self.r > bm.max(self.r) * 0.1, self.r, 0)
        center_phase = bm.zeros(2,)
        center_phase[0] = bm.angle(bm.sum(exppos_x * r))
        center_phase[1] = bm.angle(bm.sum(exppos_y * r))

        center_pos = bm.zeros(2,)
        center_pos = bm.matmul(self.coor_transform_inv, center_phase)/(self.ratio)
        # print(center_pos.shape)
        
        Candidate_center = self.Candidate_center + center_pos 
        distances = bm.linalg.norm(Candidate_center - pos, axis=1)

        # 找到最小距离的点的索引
        closest_index = bm.argmin(distances)

        # 找到最近的点
        self.center_pos = Candidate_center[closest_index]
        # self.center_pos = center_pos
        self.center_phase.value = center_phase

        d = bm.asarray(self.center_pos) - self.value_bump
        delta_x = d[:, 0]
        delta_y = d[:, 1]
        dis = bm.sqrt(delta_x ** 2 + delta_y ** 2)
        self.bump = self.A * bm.exp(-bm.square(dis / self.a))


    @property
    def derivative(self):
        du = (
            lambda u, t, Irec: (
                -u
                + Irec
                + self.input
                - self.v
            )
            / self.tau
        )
        dv = lambda v, t: (-v + self.m * self.u) / self.tau_v
        return bp.JointEq([du, dv])

    def reset_state(self):
        self.r.value = bm.Variable(bm.zeros(self.num))
        self.u.value = bm.Variable(bm.zeros(self.num))
        self.v.value = bm.Variable(bm.zeros(self.num))
        self.input.value = bm.Variable(bm.zeros(self.num))

    def update(self, Animal_location, HD_activity, ThetaModulator, HD_truth):
        self.get_center(Animal_location)
        # input_conjG = self.input_by_conjG(pos, hd, input_stre)
        input_conjG = self.input_by_conjG_new(Animal_location, HD_activity, ThetaModulator, HD_truth)
        
        self.input = input_conjG
        
        Irec = bm.matmul(self.conn_mat, self.r) + self.noise_stre * bm.random.randn(
            (self.num)
        )
        # Update neural state
        u, v = self.integral(self.u, self.v, bp.share["t"], Irec, bm.dt)
        self.u.value = bm.where(u > 0, u, 0)
        self.v.value = v
        r1 = bm.square(self.u)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = self.g*r1 / r2



def calculate_inst_speed(directions, samples_per_sec):
    diff_dist = np.diff(directions.flatten())
    # consider the periodic boundary condition that is, if diff > pi, then diff = diff - 2*pi
    # if diff < -pi, then diff = diff + 2*pi
    diff_dist = np.where(diff_dist > np.pi, diff_dist - 2 * np.pi, diff_dist)
    diff_dist = np.where(diff_dist < -np.pi, diff_dist + 2 * np.pi, diff_dist)
    inst_speed = diff_dist * samples_per_sec
    # insert the first element the same as the second element
    inst_speed = np.insert(inst_speed, 0, 0)
    return inst_speed


def create_directory_if_not_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)




def circle_period(d):
    d = np.where(d > np.pi, d - 2 * np.pi, d)
    d = np.where(d < -np.pi, d + 2 * np.pi, d)
    return d


def traj(x0, v, T):
    x = []
    xt = x0
    for i in range(T):
        xt = xt + v * bm.dt
        if xt > np.pi:
            xt -= 2 * np.pi
        if xt < -np.pi:
            xt += 2 * np.pi
        x.append(xt)
    return np.array(x)

def calculate_inst_speed(directions, samples_per_sec):
    diff_dist = np.diff(directions.flatten())
    # consider the periodic boundary condition that is, if diff > pi, then diff = diff - 2*pi
    # if diff < -pi, then diff = diff + 2*pi
    diff_dist = np.where(diff_dist > np.pi, diff_dist - 2 * np.pi, diff_dist)
    diff_dist = np.where(diff_dist < -np.pi, diff_dist + 2 * np.pi, diff_dist)
    inst_speed = diff_dist * samples_per_sec
    # insert the first element the same as the second element
    inst_speed = np.insert(inst_speed, 0, 0)
    return inst_speed


def create_directory_if_not_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def period(data):
    # 计算傅里叶变换
    fft_x = np.fft.fft(data)
    frequencies = np.fft.fftfreq(len(data), d=(bm.dt))

    # 仅使用正频率部分
    positive_frequencies = frequencies[np.where(frequencies >= 0)]
    positive_fft_x = np.abs(fft_x[np.where(frequencies >= 0)])

    # 找到最大频率分量
    dominant_frequency = positive_frequencies[np.argmax(positive_fft_x)]
    dominant_period = 1 / dominant_frequency

    # 打印结果
    print(f"Dominant Frequency: {dominant_frequency} Hz")
    print(f"Dominant Period: {dominant_period} ")


def circle_period(d):
    d = np.where(d > np.pi, d - 2 * np.pi, d)
    d = np.where(d < -np.pi, d + 2 * np.pi, d)
    return d


def traj(x0, v, T):
    x = []
    xt = x0
    for i in range(T):
        xt = xt + v * bm.dt
        if xt > np.pi:
            xt -= 2 * np.pi
        if xt < -np.pi:
            xt += 2 * np.pi
        x.append(xt)
    return np.array(x)



def straight_line(x0, v, angle, T):
    x = []
    y = []
    xt = x0
    yt = x0
    for i in range(T):
        # xt = xt + v * bm.cos(angle) * bm.dt
        # yt = yt + v * bm.sin(angle) * bm.dt
        xt = xt + v * np.cos(angle) 
        yt = yt + v * np.sin(angle) 
        x.append(xt)
        y.append(yt)
    Animal_location = np.array([x, y])
    return Animal_location