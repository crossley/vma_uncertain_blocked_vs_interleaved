from imports import *
from util_func import *

if __name__ == "__main__":

    dp = load_data()

    # NOTE: individual subject plots
    # make_fig_per_subject(dp)

    # NOTE: Exclude ppts that have aberrant movements
    subs_exc = [13, 14, 18, 19, 23, 32]

    dp = dp[~np.isin(dp["subject"], subs_exc)]

    # NOTE: average over subjects
    dpp = dp.groupby(["condition", "trial", "phase", "su_prev"],
                     observed=True)[[
                         "emv", "delta_emv", "movement_error",
                         "movement_error_prev", "rotation"
                     ]].mean().reset_index()

    make_fig_group_per_trial(dpp)

    # NOTE:
    # ph = 2
    # dpp = dp[(dp["condition"] != "interleaved") & (dp["phase"] == ph)].copy()
    # fit_regression(dpp)

    # dpp = dp[(dp["condition"] == "interleaved") & (dp["phase"] == ph)].copy()
    # fit_regression(dpp)

    # NOTE: fit and inspect state-space model
    # fit_ss_model(dp)
    # inspect_results_ss(dp)

    # TODO: Delta emv vs movement error plots
    dpp = dp.copy()

    dpp = dpp[dpp["phase"].isin([2])]

    # replace condition names for plotting with "Blocked" and "Interleaved"
    dpp["condition"] = dpp["condition"].replace({
        "Blocked - High Low": "Blocked",
        "Blocked - Low High": "Blocked",
        "interleaved": "Interleaved"
    }).copy()

    # group movement_error_prev into bins and plot delta_emv vs movement_error_prev
    bin_edges = np.arange(-30, 11, 3)
    dpp["me_prev_bin"] = pd.cut(dpp["movement_error_prev"],
                                bins=bin_edges,
                                labels=False)
    dpp_grouped = dpp.groupby(
        ["condition", "subject", "su_prev",
         "me_prev_bin"])["delta_emv"].mean().reset_index()

    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(16, 6))
    for i, cnd in enumerate(dpp_grouped["condition"].unique()):
        dcnd = dpp_grouped[dpp_grouped["condition"] == cnd].copy()
        sns.pointplot(data=dcnd,
                      x="me_prev_bin",
                      y="delta_emv",
                      hue="su_prev",
                      marker="o",
                      ax=ax[0, i])
        ax[0, i].set_title(f"Condition: {cnd}")
        ax[0, i].set_xlabel("Binned Movement Error Previous")
        ax[0, i].set_ylabel("Delta EMV")
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax[0, i].set_xticklabels([f"{bc:.2f}" for bc in bin_centers])
        ax[0, i].legend()
        ax[0, i].axvline(x=np.digitize(0, bin_edges) - 1.5,
                         color='black',
                         linestyle='--',
                         alpha=0.3)
        ax[0, i].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("../figures/delta_emv_vs_movement_error_prev_phase2.png")
    plt.close()

    # TODO:
    # [x] connect lines for within-subject plots
    # [x] write model fit figure to file
    # [x] make delta_emv vs movement_error plots
    # [ ] formaly analyse delta_emv vs movement_error plots
    # [ ] build model with dynamic learning rate for model comparison
    # [ ] fit KF models
    # [ ] wtf is with the bias in reaches --- where is zero?
