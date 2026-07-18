# https://github.com/cck56/scurve
# 授权协议https://github.com/cck56/scurve/blob/main/LICENSE
# 修改点：由c语言翻译为python
import math
from enum import Enum

# 常量定义
EPSILON = 1e-6
MAX_DIST_ERROR = 6.0
NEWTON_ITERATIONS = 10
NEWTON_TOLERANCE = 1e-3
NEWTON_RELAX_FACTOR = 1.5
VP_GUESS_ITERATIONS = 5
ON_END_RESET_VAJ = True

class ScurveState(Enum):
    CURVE_IDLE = 0
    CURVE_CONF = 1
    CURVE_BUSY = 2
    CURVE_ONEND = 3

class ScurveReturnStatus(Enum):
    CURVE_SUCCESS = 0
    CURVE_INVALID_PARAMS = 1
    CURVE_INVALID_STATE = 2

class SegmentState:
    __slots__ = ('x0', 'v0', 'a0', 'j0', 't0', 'tEnd', 'segmentIndex', 'next')
    
    def __init__(self):
        self.x0 = 0.0
        self.v0 = 0.0
        self.a0 = 0.0
        self.j0 = 0.0
        self.t0 = 0.0
        self.tEnd = 0.0
        self.segmentIndex = 0
        self.next = None

class ScurveParam:
    __slots__ = ('sign', 'dist', 's1_end', 's2_end', 's3_end', 's4_end', 's5_end', 's6_end', 'totalTime')
    
    def __init__(self):
        self.sign = 1.0
        self.dist = 0.0
        self.s1_end = 0.0
        self.s2_end = 0.0
        self.s3_end = 0.0
        self.s4_end = 0.0
        self.s5_end = 0.0
        self.s6_end = 0.0
        self.totalTime = 0.0

class Scurve:
    def __init__(self, sampleTime_s):
        self.state = ScurveState.CURVE_IDLE
        self.Ts = sampleTime_s
        self.pos_target = 0.0
        self.pos_out = 0.0
        self.vel_out = 0.0
        self.acc_out = 0.0
        self.jerk_out = 0.0
        self.vel_start = 0.0
        self.vel_stop = 0.0
        self.vel_max = 0.0
        self.acc_max = 0.0
        self.dec_max = 0.0
        self.jerk_max = 0.0
        self.vel_peak = 0.0
        self.vel_start_clip = 0.0
        self.profile_ticks = 0
        self.segStateIndex = 0
        self.segState = [SegmentState() for _ in range(7)]
        self.param = ScurveParam()
        self._segments = []

    # 辅助函数
    @staticmethod
    def integrate_segment1357(x0, v0, a0, j, dt):
        a_local = a0 + j * dt
        v_local = v0 + a0 * dt + 0.5 * j * (dt * dt)
        x_local = x0 + v0 * dt + 0.5 * a0 * (dt * dt) + (j * (dt * dt * dt)) / 6.0
        return x_local, v_local, a_local

    @staticmethod
    def integrate_segment26(x0, v0, a0, dt):
        a_local = a0
        v_local = v0 + a0 * dt
        x_local = x0 + v0 * dt + 0.5 * a0 * (dt * dt)
        return x_local, v_local, a_local

    @staticmethod
    def integrate_segment4(x0, v0, dt):
        v_local = v0
        x_local = x0 + v0 * dt
        return x_local, v_local

    def compute_scurve_distance(self, Vs, Ve, A, J):
        result = type('', (), {})()
        dv = Ve - Vs
        
        if abs(dv) < EPSILON:
            result.distance = 0.0
            result.time = 0.0
            result.Tj = 0.0
            result.Ta = 0.0
            return result
        
        dv_threshold = (A * A) / J
        if dv > dv_threshold + EPSILON:
            Tj = A / J
            Tflat = (dv - A * Tj) / A
            Ta = 2.0 * Tj + Tflat
            dist_acc = Vs * Ta + 0.5 * dv * Ta
            
            result.distance = dist_acc
            result.time = Ta
            result.Tj = Tj
            result.Ta = Ta
        else:
            Tshort = 2.0 * math.sqrt(dv / J)
            dist_acc = Vs * Tshort + (dv * Tshort / 2.0)
            
            result.distance = dist_acc
            result.time = Tshort
            result.Tj = 0.5 * Tshort
            result.Ta = Tshort
        
        return result

    def scurve_findVpeakClosedForm(self, Vs, Vm, Ve, A, Dec, J, D):
        Vp_trap_acc = Vs + (A * A) / J
        Vp_trap_dec = Ve + (Dec * Dec) / J
        Vp_min = max(Vs, Ve)
        
        # 计算候选峰值速度
        C2 = 0.5 * (1.0/A + 1.0/Dec)
        C1 = (A + Dec) / (2.0 * J)
        C0 = -0.5 * (Vs*Vs/A + Ve*Ve/Dec) + (A*Vs + Dec*Ve)/(2.0*J) - D
        disc = C1 * C1 - 4.0 * C2 * C0
        disc = max(disc, 0.0)
        Vp_candidate = (-C1 + math.sqrt(disc)) / (2.0 * C2)
        
        if Vp_candidate >= Vp_trap_acc and Vp_candidate >= Vp_trap_dec:
            return Vp_candidate
        
        Vp = max(Vp_min, Vp_candidate)
        if Vp - Vp_min < EPSILON:
            Vp = Vp_min + 1e-3
            
        for _ in range(VP_GUESS_ITERATIONS):
            localAccTri = (Vp < Vp_trap_acc)
            localDecTri = (Vp < Vp_trap_dec)
            
            f_val = 0.0
            df = 0.0
            
            # 加速段
            if localAccTri:
                diff = Vp - Vs
                sqrt_term = math.sqrt(diff / J)
                f_val += (Vp + Vs) * sqrt_term
                df += sqrt_term + (Vp + Vs) / (2.0 * math.sqrt(J * diff))
            else:
                diff = Vp - Vs
                term = diff / A + A / J
                f_val += 0.5 * (Vp + Vs) * term
                df += Vp / A + 0.5 * (A / J)
                
            # 减速段
            if localDecTri:
                diff = Vp - Ve
                sqrt_term = math.sqrt(diff / J)
                f_val += (Vp + Ve) * sqrt_term
                df += sqrt_term + (Vp + Ve) / (2.0 * math.sqrt(J * diff))
            else:
                diff = Vp - Ve
                term = diff / Dec + Dec / J
                f_val += 0.5 * (Vp + Ve) * term
                df += Vp / Dec + 0.5 * (Dec / J)
                
            f_val -= D
            newVp = Vp - f_val / df
            
            if newVp < Vp_min:
                newVp = 0.5 * (Vp + Vp_min)
            Vp = newVp
            
        return Vp

    def calc_param(self):
        dist = self.pos_target - self.pos_out
        self.param.dist = dist
        self.param.sign = 1.0 if dist >= 0.0 else -1.0
        D = abs(dist)
        Vs = self.vel_start
        Ve = self.vel_stop
        Vm = self.vel_max
        A = self.acc_max
        Dec = self.dec_max
        J = self.jerk_max
        
        # 计算加速段和减速段
        accPart = self.compute_scurve_distance(Vs, Vm, A, J)
        decPart = self.compute_scurve_distance(Ve, Vm, Dec, J)
        
        s_acc = accPart.distance
        s_dec = decPart.distance
        t_acc = accPart.time
        t_dec = decPart.time
        s_sum = s_acc + s_dec
        
        s_cruise = 0.0
        t_cruise = 0.0
        
        if s_sum < D:
            s_cruise = D - s_sum
            t_cruise = s_cruise / Vm
            self.vel_peak = Vm
            self.vel_start_clip = Vs
        else:
            vstart_clipped = False
            if Vs > Ve:
                decPart = self.compute_scurve_distance(Ve, Vs, Dec, J)
                if decPart.distance > D:
                    dv = Vs - Ve
                    dv_threshold = (Dec * Dec) / J
                    if dv <= dv_threshold + EPSILON:
                        # 三角形减速段
                        half_term = (math.sqrt(J) * D) / 2.0
                        delta_val = half_term**2 + (8.0 * Ve**3) / 27.0
                        w = math.cbrt(half_term + math.sqrt(delta_val)) + \
                            math.cbrt(half_term - math.sqrt(delta_val))
                        
                        for _ in range(NEWTON_ITERATIONS):
                            f_val = w**3 + 2.0*Ve*w - math.sqrt(J)*D
                            if abs(f_val) < NEWTON_TOLERANCE:
                                break
                            df = 3.0*w*w + 2.0*Ve
                            dw = f_val / df
                            w -= NEWTON_RELAX_FACTOR * dw
                            
                        x = w*w
                        VsCandidate = Ve + x
                    else:
                        # 梯形减速段
                        C = Vs + Ve
                        Vdelta = Vs - Ve
                        E = Vdelta + (Dec*Dec)/J
                        disc_val = (C + E)**2 - 4.0*(C*E - 2.0*Dec*D)
                        disc_val = max(disc_val, 0.0)
                        delta_trap = ((C + E) - math.sqrt(disc_val)) / 2.0
                        VsCandidate = Vs - delta_trap
                        
                        if VsCandidate - Ve < dv_threshold:
                            half_term = (math.sqrt(J)*D)/2.0
                            delta_val = half_term**2 + (8.0*Ve**3)/27.0
                            w = math.cbrt(half_term + math.sqrt(delta_val)) + \
                                math.cbrt(half_term - math.sqrt(delta_val))
                            for _ in range(NEWTON_ITERATIONS):
                                f_val = w**3 + 2.0*Ve*w - math.sqrt(J)*D
                                if abs(f_val) < NEWTON_TOLERANCE:
                                    break
                                df = 3.0*w*w + 2.0*Ve
                                dw = f_val / df
                                w -= NEWTON_RELAX_FACTOR * dw
                            x = w*w
                            VsCandidate = Ve + x
                    
                    VsCandidate = max(min(VsCandidate, Vs), Ve)
                    decPart = self.compute_scurve_distance(Ve, VsCandidate, Dec, J)
                    distError = decPart.distance - D
                    
                    if abs(distError) <= MAX_DIST_ERROR:
                        accPart.distance = 0.0
                        accPart.time = 0.0
                        accPart.Tj = 0.0
                        vstart_clipped = True
                        Vs = VsCandidate
                        
            elif Ve > Vs:
                accPart = self.compute_scurve_distance(Vs, Ve, A, J)
                if accPart.distance > D:
                    dv = Ve - Vs
                    dv_threshold = A * A / J
                    if dv <= dv_threshold + EPSILON:
                        # 三角形加速段
                        a_coef = 4.0 * Vs
                        p_coef = -(4.0 * Vs**2) / 3.0
                        q_coef = -(16.0 * Vs**3) / 27.0 - (J * D**2)
                        half_q = q_coef / 2.0
                        disc = half_q**2 + (p_coef/3.0)**3
                        tol = 1e-6 * abs(half_q**2)
                        if disc < 0.0 and abs(disc) < tol:
                            disc = 0.0
                            
                        sqrt_val = math.sqrt(disc)
                        y = math.cbrt(-half_q + sqrt_val) + math.cbrt(-half_q - sqrt_val)
                        delta = y - a_coef/3.0
                        VeCandidate = Vs + delta
                    else:
                        # 梯形加速段
                        b = 2.0 * Vs + (A**2) / J
                        c_term = (2.0 * Vs * A**2) / J - 2.0 * A * D
                        disc = b**2 - 4.0 * c_term
                        disc = max(disc, 0.0)
                        sqrt_disc = math.sqrt(disc)
                        delta1 = (-b + sqrt_disc) / 2.0
                        delta2 = (-b - sqrt_disc) / 2.0
                        delta = max(delta1, delta2)
                        VeCandidate = Vs + delta
                        
                        if delta < dv_threshold:
                            a_coef = 4.0 * Vs
                            p_coef = -(4.0 * Vs**2) / 3.0
                            q_coef = -(16.0 * Vs**3) / 27.0 - (J * D**2)
                            half_q = q_coef / 2.0
                            disc = half_q**2 + (p_coef/3.0)**3
                            tol = 1e-6 * abs(half_q**2)
                            if disc < 0.0 and abs(disc) < tol:
                                disc = 0.0
                            sqrt_val = math.sqrt(disc)
                            y = math.cbrt(-half_q + sqrt_val) + math.cbrt(-half_q - sqrt_val)
                            delta = y - a_coef/3.0
                            VeCandidate = Vs + delta
                    
                    VeCandidate = max(min(VeCandidate, Ve), Vs)
                    accPart = self.compute_scurve_distance(Vs, VeCandidate, A, J)
                    distError = accPart.distance - D
                    
                    if abs(distError) <= MAX_DIST_ERROR:
                        decPart.distance = 0.0
                        decPart.time = 0.0
                        decPart.Tj = 0.0
                        vstart_clipped = True
                        
            if not vstart_clipped:
                Vp = self.scurve_findVpeakClosedForm(Vs, Vm, Ve, A, Dec, J, D)
                if Vp <= Ve:
                    Ve = Vp
                    Vm = Vp
                    accPart = self.compute_scurve_distance(Vs, Vm, A, J)
                    decPart.distance = 0.0
                    decPart.time = 0.0
                    decPart.Tj = 0.0
                elif Vp < Vs:
                    Vs = Vp
                    Vm = Vp
                    decPart = self.compute_scurve_distance(Vs, Vm, Dec, J)
                    accPart.distance = 0.0
                    accPart.time = 0.0
                    accPart.Tj = 0.0
                else:
                    Vm = Vp
                    accPart = self.compute_scurve_distance(Vs, Vm, A, J)
                    decPart = self.compute_scurve_distance(Ve, Vm, Dec, J)
            
            self.vel_peak = Vm
            self.vel_start_clip = Vs
            s_acc = accPart.distance
            s_dec = decPart.distance
            t_acc = accPart.time
            t_dec = decPart.time
            
            if s_acc + s_dec < D:
                s_cruise = D - (s_acc + s_dec)
                t_cruise = s_cruise / Vm
            else:
                s_cruise = 0.0
                t_cruise = 0.0
        
        # 计算各段时间
        T1 = accPart.Tj
        T2 = t_acc - 2.0 * accPart.Tj
        T3 = accPart.Tj
        T4 = t_cruise
        T5 = decPart.Tj
        T6 = t_dec - 2.0 * decPart.Tj
        T7 = decPart.Tj
        
        # 存储参数
        self.param.s1_end = T1
        self.param.s2_end = self.param.s1_end + T2
        self.param.s3_end = self.param.s2_end + T3
        self.param.s4_end = self.param.s3_end + T4
        self.param.s5_end = self.param.s4_end + T5
        self.param.s6_end = self.param.s5_end + T6
        self.param.totalTime = self.param.s6_end + T7

    def segment_eval(self, seg, t_sec):
        dt = t_sec - seg.t0
        
        if seg.segmentIndex in (1, 3, 5, 7):
            x, v, a = self.integrate_segment1357(seg.x0, seg.v0, seg.a0, seg.j0, dt)
            j = seg.j0
            return x, v, a, j
        
        elif seg.segmentIndex in (2, 6):
            x, v, a = self.integrate_segment26(seg.x0, seg.v0, seg.a0, dt)
            j = 0.0
            return x, v, a, j
        
        elif seg.segmentIndex == 4:
            x, v = self.integrate_segment4(seg.x0, seg.v0, dt)
            j = 0.0
            a = 0.0
            return x, v, a, j
        
        return 0.0, 0.0, 0.0, 0.0

    def run(self):
        if self.state == ScurveState.CURVE_IDLE:
            self.profile_ticks = 0
            return self.state
        
        if self.state == ScurveState.CURVE_CONF:
            self.calc_param()
            
            segEnd = [
                self.param.s1_end,
                self.param.s2_end,
                self.param.s3_end,
                self.param.s4_end,
                self.param.s5_end,
                self.param.s6_end,
                self.param.totalTime
            ]
            
            # 构建段链表
            self._segments = []
            tCurrent = 0.0
            segCount = 0
            
            # 创建第一段
            firstSeg = self.segState[segCount]
            segCount += 1
            firstSeg.segmentIndex = 1
            firstSeg.t0 = 0.0
            firstSeg.tEnd = segEnd[0]
            firstSeg.x0 = self.pos_out
            firstSeg.v0 = self.param.sign * self.vel_start_clip
            firstSeg.a0 = 0.0
            firstSeg.j0 = self.param.sign * self.jerk_max
            self._segments.append(firstSeg)
            
            prevSeg = firstSeg
            tCurrent = segEnd[0]
            
            # 创建后续段
            for i in range(1, 7):
                tEnd = segEnd[i]
                duration = tEnd - tCurrent
                
                if duration > EPSILON:
                    seg = self.segState[segCount]
                    segCount += 1
                    seg.segmentIndex = i + 1
                    seg.t0 = tCurrent
                    seg.tEnd = tEnd
                    seg.next = None
                    
                    # 计算上一段结束状态
                    xEnd, vEnd, aEnd, jEnd = self.segment_eval(prevSeg, prevSeg.tEnd)
                    seg.x0 = xEnd
                    seg.v0 = vEnd
                    seg.a0 = aEnd
                    
                    # 设置初始加加速度
                    if seg.segmentIndex == 2:
                        seg.j0 = 0.0
                    elif seg.segmentIndex == 3:
                        seg.j0 = -self.param.sign * self.jerk_max
                    elif seg.segmentIndex == 4:
                        seg.j0 = 0.0
                    elif seg.segmentIndex == 5:
                        seg.j0 = -self.param.sign * self.jerk_max
                    elif seg.segmentIndex == 6:
                        seg.j0 = 0.0
                    elif seg.segmentIndex == 7:
                        seg.j0 = self.param.sign * self.jerk_max
                    
                    prevSeg.next = seg
                    self._segments.append(seg)
                    prevSeg = seg
                
                tCurrent = tEnd
            
            self.segStateIndex = 0
            self.profile_ticks = 0
            self.state = ScurveState.CURVE_BUSY
        
        if self.state == ScurveState.CURVE_BUSY:
            t_sec = self.profile_ticks * self.Ts
            self.profile_ticks += 1
            seg = self._segments[self.segStateIndex]
            
            if t_sec >= seg.tEnd:
                if seg.next is None:
                    # 结束处理
                    x, v, a, j = self.segment_eval(seg, self.param.totalTime)
                    if self.param.sign * (x - self.pos_target) > 0.0:
                        x = self.pos_target
                    self.pos_out = x
                    self.vel_out = v
                    self.acc_out = a
                    self.jerk_out = j
                    self.state = ScurveState.CURVE_ONEND
                    return self.state
                
                # 移动到下一段
                self.segStateIndex += 1
                seg = self._segments[self.segStateIndex]
            
            # 计算当前状态
            x, v, a, j = self.segment_eval(seg, t_sec)
            
            # 检查是否超过目标位置
            if self.param.sign * (x - self.pos_target) >= 0.0:
                x = self.pos_target
                self.state = ScurveState.CURVE_ONEND
            
            self.pos_out = x
            self.vel_out = v
            self.acc_out = a
            self.jerk_out = j
            return self.state
        
        if self.state == ScurveState.CURVE_ONEND:
            self.pos_out = self.pos_target
            if ON_END_RESET_VAJ:
                self.vel_out = 0.0
                self.acc_out = 0.0
                self.jerk_out = 0.0
            self.state = ScurveState.CURVE_IDLE
            return self.state
        
        return self.state

    # 公共接口方法
    def startProfile(self):
        if self.state != ScurveState.CURVE_IDLE:
            return ScurveReturnStatus.CURVE_INVALID_STATE
            
        if self.vel_max <= self.vel_start or self.vel_max <= self.vel_stop:
            return ScurveReturnStatus.CURVE_INVALID_PARAMS
            
        if self.vel_max <= 0.0 or self.acc_max <= 0.0 or self.dec_max <= 0.0 or self.jerk_max <= 0.0:
            return ScurveReturnStatus.CURVE_INVALID_PARAMS
            
        self.state = ScurveState.CURVE_CONF
        return ScurveReturnStatus.CURVE_SUCCESS

    def getState(self):
        return self.state

    def getOutputs(self):
        return self.pos_out, self.vel_out, self.acc_out, self.jerk_out

    def setSampleTime(self, sampleTime_s):
        if self.state == ScurveState.CURVE_IDLE:
            self.Ts = sampleTime_s

    def setPositionTarget(self, pos_target):
        if self.state == ScurveState.CURVE_IDLE:
            self.pos_target = pos_target

    def setPositionOutput(self, pos_out):
        if self.state == ScurveState.CURVE_IDLE:
            self.pos_out = pos_out

    def setVelocityStart(self, vel_start):
        if self.state == ScurveState.CURVE_IDLE:
            self.vel_start = vel_start

    def setVelocityStop(self, vel_stop):
        if self.state == ScurveState.CURVE_IDLE:
            self.vel_stop = vel_stop

    def setVelocityMax(self, vel):
        if self.state == ScurveState.CURVE_IDLE:
            self.vel_max = vel

    def setAccelerationMax(self, acc):
        if self.state == ScurveState.CURVE_IDLE:
            self.acc_max = acc

    def setDecelerationMax(self, dec):
        if self.state == ScurveState.CURVE_IDLE:
            self.dec_max = dec

    def setJerkMax(self, jerk):
        if self.state == ScurveState.CURVE_IDLE:
            self.jerk_max = jerk

    def getPositionOutput(self):
        return self.pos_out

    def getVelocityOutput(self):
        return self.vel_out

    def getAccelerationOutput(self):
        return self.acc_out

    def getJerkOutput(self):
        return self.jerk_out

    def getProfileTick(self):
        return self.profile_ticks