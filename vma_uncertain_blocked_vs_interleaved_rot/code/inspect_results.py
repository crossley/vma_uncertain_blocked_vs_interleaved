from imports import *
from util_func import *

if __name__ == "__main__":

    dp = load_data()

    #    # NOTE: individual subject plots
    #    # make_fig_per_subject(dp)
    #
    # NOTE: Exclude ppts that have aberrant movements
    subs_exc = [5, 10, 17, 22, 30, 31, 35, 37, 40, 55, 56, 57, 62]

    dp = dp[~np.isin(dp["subject"], subs_exc)]

    # # NOTE: average over subjects
    # dpp = dp.groupby(["condition", "trial", "phase", "su_prev"],
    #                  observed=True)[[
    #                      "emv", "delta_emv", "movement_error",
    #                      "movement_error_prev", "rotation"
    #                  ]].mean().reset_index()

    # make_fig_group_per_trial(dpp)

    # NOTE:
    # ph = 2
    # dpp = dp[(dp["condition"] != "interleaved") & (dp["phase"] == ph)].copy()
    # fit_regression(dpp)

    # dpp = dp[(dp["condition"] == "interleaved") & (dp["phase"] == ph)].copy()
    # fit_regression(dpp)

    # NOTE: fit and inspect state-space model
    # fit_ss_model(dp)
    # inspect_results_ss(dp)

    # NOTE: Delta emv vs movement error plots
    # make_fig_dela_emv_vs_movement_error_prev(dp)

    # TODO:
    # [x] connect lines for within-subject plots
    # [x] write model fit figure to file
    # [x] make delta_emv vs movement_error plots
    # [x] build model with dynamic learning rate for model comparison
    # [x] wtf is with the bias in reaches --- where is zero?
    # [ ] formaly analyse delta_emv vs movement_error plots

    psp_ss()
