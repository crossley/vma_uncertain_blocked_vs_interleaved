import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Classic 1D Kalman filter with known R_t
# ---------------------------------------------------------

def simulate_kf_known_R(
    perturbation,
    R_seq,
    Q=1e-3,
    x0=0.0,
    P0=1.0,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    T = len(perturbation)
    x_hat = np.zeros(T + 1)
    P = np.zeros(T + 1)
    K = np.zeros(T)
    y = np.zeros(T)
    e = np.zeros(T)

    x_hat[0] = x0
    P[0] = P0

    for t in range(T):
        # Predict
        x_pred = x_hat[t]
        P_pred = P[t] + Q

        # Observation
        y[t] = perturbation[t] + rng.normal(0, np.sqrt(R_seq[t]))

        # Kalman gain using true R_t
        K[t] = P_pred / (P_pred + R_seq[t])

        # Update
        e[t] = y[t] - x_pred
        x_hat[t + 1] = x_pred + K[t] * e[t]
        P[t + 1] = (1 - K[t]) * P_pred

    return {
        "x_hat": x_hat,
        "P": P,
        "K": K,
        "y": y,
        "e": e,
    }


# ---------------------------------------------------------
# 2. Adaptive "single-R" model
# ---------------------------------------------------------

def simulate_kf_adaptive_R(
    perturbation,
    R_true_seq,
    Q=1e-3,
    x0=0.0,
    P0=1.0,
    R0=1.0,
    alpha_R=0.05,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    T = len(perturbation)
    x_hat = np.zeros(T + 1)
    P = np.zeros(T + 1)
    K = np.zeros(T)
    y = np.zeros(T)
    e = np.zeros(T)
    R_hat = np.zeros(T + 1)

    x_hat[0] = x0
    P[0] = P0
    R_hat[0] = R0

    for t in range(T):
        # Predict
        x_pred = x_hat[t]
        P_pred = P[t] + Q

        # Observation using true (experiment-side) R_t
        y[t] = perturbation[t] + rng.normal(0, np.sqrt(R_true_seq[t]))

        # Kalman gain uses learner's R_hat
        K[t] = P_pred / (P_pred + R_hat[t])

        # Update state
        e[t] = y[t] - x_pred
        x_hat[t + 1] = x_pred + K[t] * e[t]
        P[t + 1] = (1 - K[t]) * P_pred

        # Update R_hat from squared residuals
        R_hat[t + 1] = (1 - alpha_R) * R_hat[t] + alpha_R * (e[t] ** 2)

    return {
        "x_hat": x_hat,
        "P": P,
        "K": K,
        "y": y,
        "e": e,
        "R_hat": R_hat,
    }


# ---------------------------------------------------------
# 3. Define schedules: blocked low, blocked high, interleaved
# ---------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    T = 200
    true_perturbation = np.ones(T) * 15.0  # e.g., 15° rotation

    R_low = 1.0
    R_high = 9.0

    # Blocked low: all low noise
    R_blocked_low = np.ones(T) * R_low

    # Blocked high: all high noise
    R_blocked_high = np.ones(T) * R_high

    # Interleaved: mix of low/high, same counts as one low + one high block
    R_interleaved = np.array([R_low] * (T // 2) + [R_high] * (T - T // 2))
    rng.shuffle(R_interleaved)

    # Classic KF
    classic_blocked_low = simulate_kf_known_R(true_perturbation, R_blocked_low, rng=rng)
    classic_blocked_high = simulate_kf_known_R(true_perturbation, R_blocked_high, rng=rng)
    classic_interleaved = simulate_kf_known_R(true_perturbation, R_interleaved, rng=rng)

    # Adaptive KF
    adapt_blocked_low = simulate_kf_adaptive_R(true_perturbation, R_blocked_low, rng=rng)
    adapt_blocked_high = simulate_kf_adaptive_R(true_perturbation, R_blocked_high, rng=rng)
    adapt_interleaved = simulate_kf_adaptive_R(true_perturbation, R_interleaved, rng=rng)

    # ---------------------------------------------------------
    # 4. Simple figure: one panel per schedule
    #   • Left subplot = Classic KF
    #   • Right subplot = Adaptive KF
    #   • Each subplot shows: blocked low, blocked high, interleaved
    # ---------------------------------------------------------

    plt.figure(figsize=(12, 5))

    # ----------------------
    # Classic KF (left panel)
    # ----------------------
    ax1 = plt.subplot(1, 2, 1)

    ax1.plot(classic_blocked_low["x_hat"], label="Blocked Low", color="green")
    ax1.plot(classic_blocked_high["x_hat"], label="Blocked High", color="orange")
    ax1.plot(classic_interleaved["x_hat"], label="Interleaved", color="purple")

    ax1.axhline(15, linestyle=":", color="gray")
    ax1.set_title("Classic KF (known R)")
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("Estimate x̂")
    ax1.legend()

    # -------------------------
    # Adaptive KF (right panel)
    # -------------------------
    ax2 = plt.subplot(1, 2, 2)

    ax2.plot(adapt_blocked_low["x_hat"], label="Blocked Low", color="green")
    ax2.plot(adapt_blocked_high["x_hat"], label="Blocked High", color="orange")
    ax2.plot(adapt_interleaved["x_hat"], label="Interleaved", color="purple")

    ax2.axhline(15, linestyle=":", color="gray")
    ax2.set_title("Adaptive KF (learned R̂)")
    ax2.set_xlabel("Trial")
    ax2.legend()

    plt.suptitle("Comparison of Noise Schedules Across Models")
    plt.tight_layout()
    plt.show()


    def compute_trial_learning_rate(sim):
        """
        Approximate trial-by-trial learning rate:
        alpha_t = (x_{t+1} - x_t) / e_t
        """
        dx = sim["x_hat"][1:] - sim["x_hat"][:-1]   # change in estimate
        e = sim["e"]                                # prediction error
        alpha = np.full_like(e, np.nan, dtype=float)

        nonzero = np.abs(e) > 1e-8                  # avoid division by ~0
        alpha[nonzero] = dx[nonzero] / e[nonzero]
        return alpha

    alpha_classic_inter = compute_trial_learning_rate(classic_interleaved)
    alpha_adapt_inter   = compute_trial_learning_rate(adapt_interleaved)

    # Masks for low vs high noise trials in the interleaved schedule
    low_mask  = R_interleaved == R_low
    high_mask = R_interleaved == R_high


    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # -----------------------------
    # Classic KF: empirical alpha_t
    # -----------------------------
    ax = axes[0]

    ax.scatter(
        np.where(low_mask)[0],
        alpha_classic_inter[low_mask],
        s=15, alpha=0.5, label="Low-noise trials"
    )
    ax.scatter(
        np.where(high_mask)[0],
        alpha_classic_inter[high_mask],
        s=15, alpha=0.5, label="High-noise trials"
    )

    # Plot mean lines for each trial type
    mean_low  = np.nanmean(alpha_classic_inter[low_mask])
    mean_high = np.nanmean(alpha_classic_inter[high_mask])
    ax.axhline(mean_low,  color="C0", linestyle="-",  linewidth=2)
    ax.axhline(mean_high, color="C1", linestyle="-",  linewidth=2)

    ax.set_title("Classic KF – Interleaved")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Empirical learning rate α_t")
    ax.legend()

    # -------------------------------
    # Adaptive KF: empirical alpha_t
    # -------------------------------
    ax = axes[1]

    ax.scatter(
        np.where(low_mask)[0],
        alpha_adapt_inter[low_mask],
        s=15, alpha=0.5, label="Low-noise trials"
    )
    ax.scatter(
        np.where(high_mask)[0],
        alpha_adapt_inter[high_mask],
        s=15, alpha=0.5, label="High-noise trials"
    )

    mean_low  = np.nanmean(alpha_adapt_inter[low_mask])
    mean_high = np.nanmean(alpha_adapt_inter[high_mask])
    ax.axhline(mean_low,  color="C0", linestyle="-", linewidth=2)
    ax.axhline(mean_high, color="C1", linestyle="-", linewidth=2)

    ax.set_title("Adaptive KF – Interleaved")
    ax.set_xlabel("Trial")
    ax.legend()

    plt.suptitle("Empirical Trial-by-Trial Learning Rate by Noise Level (Interleaved)")
    plt.tight_layout()
    plt.show()

