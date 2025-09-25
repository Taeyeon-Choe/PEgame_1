import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional dependency
    njit = None


_NUMBA_AVAILABLE = njit is not None
_JIT_REG_EPS = np.float32(1e-6)


def _broadcast_matrix(mat: np.ndarray, horizon: int) -> np.ndarray:
    out = np.empty((horizon, mat.shape[0], mat.shape[1]), dtype=mat.dtype)
    for k in range(horizon):
        out[k] = mat
    return out


def _to_matrix_sequence(seq, size: int, horizon: int, *, dtype: np.dtype) -> np.ndarray:
    """Return (horizon, size, size) float32 cost matrices."""
    arr = np.asarray(seq, dtype=dtype)

    if arr.ndim == 1:
        if arr.shape[0] != size:
            raise ValueError(f"Expected {size} diagonal entries, got {arr.shape[0]}")
        mat = np.diag(arr)
        return _broadcast_matrix(mat, horizon)

    if arr.ndim == 2:
        if arr.shape == (size, size):
            return _broadcast_matrix(arr, horizon)
        if arr.shape[0] == horizon and arr.shape[1] == size:
            out = np.empty((horizon, size, size), dtype=dtype)
            for k in range(horizon):
                out[k] = np.diag(arr[k])
            return out
        raise ValueError(
            f"Unsupported 2D cost shape {arr.shape}; expected {(size, size)} or {(horizon, size)}"
        )

    if arr.ndim == 3:
        if arr.shape[0] < horizon or arr.shape[1:] != (size, size):
            raise ValueError(
                f"Cost sequence must be (>=H, {size}, {size}), got {arr.shape}"
            )
        out = np.empty((horizon, size, size), dtype=dtype)
        for k in range(horizon):
            out[k] = arr[k]
        return out

    raise ValueError("Unsupported cost specification")


def _to_terminal_matrix(QN, size: int, *, dtype: np.dtype) -> np.ndarray:
    arr = np.asarray(QN, dtype=dtype)
    if arr.ndim == 1:
        if arr.shape[0] != size:
            raise ValueError(f"QN must have {size} diagonal elements")
        return np.diag(arr)
    if arr.shape == (size, size):
        return arr.copy()
    raise ValueError("QN must be length-n diag or (n, n) matrix")


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _tvlqr_gains_numba(A_seq, B_seq, Q_seq, R_seq, QN):
        horizon = A_seq.shape[0]
        m = B_seq.shape[2]
        n = A_seq.shape[1]

        K_seq = np.zeros((horizon, m, n), dtype=A_seq.dtype)
        P = QN.copy()

        for k in range(horizon - 1, -1, -1):
            A = A_seq[k]
            B = B_seq[k]
            Qk = Q_seq[k]
            Rk = R_seq[k]

            PB = P @ B
            S = Rk + B.T @ PB
            for i in range(m):
                for j in range(i + 1, m):
                    sym = 0.5 * (S[i, j] + S[j, i])
                    S[i, j] = sym
                    S[j, i] = sym
                S[i, i] = S[i, i] + _JIT_REG_EPS

            F = (B.T @ P) @ A
            K = np.linalg.solve(S, F)
            K_seq[k] = K

            P = Qk + A.T @ P @ A - A.T @ PB @ K

        return K_seq


def _tvlqr_gains_python(A_seq, B_seq, Q_seq, R_seq, QN):
    horizon = A_seq.shape[0]
    m = B_seq.shape[2]
    n = A_seq.shape[1]

    K_seq = np.zeros((horizon, m, n), dtype=A_seq.dtype)
    P = QN.copy()

    for k in range(horizon - 1, -1, -1):
        A = A_seq[k]
        B = B_seq[k]
        Qk = Q_seq[k]
        Rk = R_seq[k]

        PB = P @ B
        S = Rk + B.T @ PB
        S = 0.5 * (S + S.T)
        for i in range(m):
            S[i, i] = S[i, i] + _JIT_REG_EPS

        F = (B.T @ P) @ A
        try:
            K = np.linalg.solve(S, F)
        except np.linalg.LinAlgError:
            K = np.linalg.lstsq(S, F, rcond=None)[0]
        K_seq[k] = K

        P = Qk + A.T @ P @ A - A.T @ PB @ K

    return K_seq


def tvlqr_gains(A_seq, B_seq, Q_seq, R_seq, QN):
    """Discrete-time finite-horizon TV-LQR gain computation (float32)."""
    A_seq = np.ascontiguousarray(np.asarray(A_seq, dtype=np.float32))
    B_seq = np.ascontiguousarray(np.asarray(B_seq, dtype=np.float32))

    if A_seq.ndim != 3 or B_seq.ndim != 3:
        raise ValueError("A_seq and B_seq must be 3-D arrays")

    horizon, n, n_col = A_seq.shape
    if n != n_col:
        raise ValueError("A_seq must be (N, n, n)")

    if B_seq.shape[0] != horizon or B_seq.shape[1] != n:
        raise ValueError("B_seq must be (N, n, m)")
    m = B_seq.shape[2]

    Q_mats = _to_matrix_sequence(Q_seq, n, horizon, dtype=np.float32)
    R_mats = _to_matrix_sequence(R_seq, m, horizon, dtype=np.float32)
    Q_mats = np.ascontiguousarray(Q_mats)
    R_mats = np.ascontiguousarray(R_mats)

    QN_mat = np.ascontiguousarray(_to_terminal_matrix(QN, n, dtype=np.float32))

    try:
        if _NUMBA_AVAILABLE:
            result = _tvlqr_gains_numba(A_seq, B_seq, Q_mats, R_mats, QN_mat)
        else:
            result = _tvlqr_gains_python(A_seq, B_seq, Q_mats, R_mats, QN_mat)
    except Exception:
        result = _tvlqr_gains_python(A_seq, B_seq, Q_mats, R_mats, QN_mat)

    return np.ascontiguousarray(result, dtype=np.float32)


def tvlqr_action(x, A_seq, B_seq, Q_seq, R_seq, QN, dv_max=None):
    """Return first-step TV-LQR impulse command, optionally saturated."""
    x_arr = np.asarray(x, dtype=np.float32).reshape(-1)
    A_arr = np.asarray(A_seq)
    if A_arr.ndim != 3 or A_arr.shape[1] != x_arr.shape[0]:
        raise ValueError("State dimension mismatch between x and A_seq")

    K_seq = tvlqr_gains(A_seq, B_seq, Q_seq, R_seq, QN)
    u = -K_seq[0] @ x_arr

    if dv_max is not None and dv_max > 0:
        norm = float(np.linalg.norm(u.astype(np.float32)))
        if norm > dv_max:
            scale = dv_max / (norm + 1e-12)
            u = scale * u

    return u.astype(np.float32)
