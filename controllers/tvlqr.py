import numpy as np


def _to_matrix_sequence(seq, size, horizon):
    """Helper: ensure each cost entry becomes a square matrix per step."""
    arr = np.asarray(seq)
    if arr.ndim == 1:
        mat = np.diag(arr)
        return [mat.copy() for _ in range(horizon)]
    if arr.ndim == 2:
        if arr.shape != (size, size):
            raise ValueError(f"Matrix must be square ({size}x{size}), got {arr.shape}")
        return [arr.copy() for _ in range(horizon)]
    if arr.ndim == 3:
        if arr.shape[0] < horizon:
            raise ValueError("Cost sequence shorter than horizon")
        mats = []
        for k in range(horizon):
            item = arr[k]
            if item.ndim == 1:
                mats.append(np.diag(item))
            elif item.shape == (size, size):
                mats.append(item)
            else:
                raise ValueError(f"Invalid cost shape at step {k}: {item.shape}")
        return mats
    raise ValueError("Unsupported cost specification")


def tvlqr_gains(A_seq, B_seq, Q_seq, R_seq, QN):
    """Discrete-time finite-horizon TV-LQR gain computation."""
    A_seq = np.asarray(A_seq, dtype=float)
    B_seq = np.asarray(B_seq, dtype=float)

    if A_seq.ndim != 3 or B_seq.ndim != 3:
        raise ValueError("A_seq and B_seq must be 3-D arrays")

    horizon = A_seq.shape[0]
    n = A_seq.shape[1]
    m = B_seq.shape[2]

    if A_seq.shape[2] != n:
        raise ValueError("A_seq must be (N, n, n)")
    if B_seq.shape[0] != horizon or B_seq.shape[1] != n or B_seq.shape[2] != m:
        raise ValueError("B_seq must be (N, n, m)")

    Q_mats = _to_matrix_sequence(Q_seq, n, horizon)
    R_mats = _to_matrix_sequence(R_seq, m, horizon)

    QN = np.asarray(QN, dtype=float)
    if QN.ndim == 1:
        QN = np.diag(QN)
    elif QN.shape != (n, n):
        raise ValueError("QN must be length-n diag or (n, n) matrix")

    K_list = [np.zeros((m, n), dtype=float) for _ in range(horizon)]
    # P stores P_{k+1} during backward iteration
    P = QN.copy()

    for k in reversed(range(horizon)):
        A = A_seq[k]
        B = B_seq[k]
        Qk = Q_mats[k]
        Rk = R_mats[k]

        S = Rk + B.T @ P @ B
        # 수치 안정화를 위한 대칭화 및 최소 리지 추가
        S = 0.5 * (S + S.T)
        S += 1e-10 * np.eye(m)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        # 표준 이산 시간 리카티 갱신식
        K = S_inv @ (B.T @ P @ A)
        K_list[k] = K
        P = Qk + A.T @ P @ A - A.T @ P @ B @ K

    return np.stack(K_list, axis=0)


def tvlqr_action(x, A_seq, B_seq, Q_seq, R_seq, QN, dv_max=None):
    """Return first-step TV-LQR impulse command, optionally saturated."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if A_seq.shape[1] != x.shape[0]:
        raise ValueError("State dimension mismatch between x and A_seq")

    K_seq = tvlqr_gains(A_seq, B_seq, Q_seq, R_seq, QN)
    u = -K_seq[0] @ x

    if dv_max is not None and dv_max > 0:
        norm = np.linalg.norm(u)
        if norm > dv_max:
            u = (dv_max / (norm + 1e-12)) * u

    return u.astype(np.float32)
