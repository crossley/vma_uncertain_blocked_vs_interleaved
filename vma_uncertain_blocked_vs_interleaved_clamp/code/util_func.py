from imports import *


def fit_ss_model(d):

    froot = "../fits/"

    # alpha_low, alpha_high, beta
    bounds = (
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
    )

    # constraints = LinearConstraint(
    #     A=[
    #         [ 0, 0, 0, ],
    #         [ 0, 0, 0, ],
    #         [ 0, 0, 0, ],
    #     ],
    #     lb=[ 0, 0, 0, ],
    #     ub=[ 0, 0, 0, ],
    # )

    # to improve your chances of finding a global minimum use higher
    # popsize (default 15), with higher mutation (default 0.5) and
    # (dithering), but lower recombination (default 0.7). this has the
    # effect of widening the search radius, but slowing convergence.
    fit_args = {
        "obj_func": obj_func,
        "sim_func": sim_func,
        "bounds": bounds,
        # "constraints": constraints,
        "disp": False,
        "maxiter": 3000,
        "popsize": 22,
        "mutation": 0.8,
        "recombination": 0.4,
        "tol": 1e-3,
        "polish": True,
        "updating": "deferred",
        "workers": -1,
    }

    for cnd in d["condition"].unique():

        dcnd = d[d["condition"] == cnd]

        for sub in dcnd["subject"].unique():

            print(sub)

            dsub = dcnd[dcnd["subject"] == sub].copy()
            dsub = dsub[dsub["phase"].isin([2])]
            dsub = dsub[["rotation", "emv", "trial", "su", "phase"]]

            rot = dsub["rotation"].to_numpy()
            x_obs = dsub["emv"].to_numpy()
            su = dsub["su"].to_numpy()
            phase = dsub["phase"].to_numpy()

            args = (rot, x_obs, su, phase)

            results = differential_evolution(
                func=fit_args["obj_func"],
                bounds=fit_args["bounds"],
                # constraints=fit_args["constraints"],
                args=args,
                disp=fit_args["disp"],
                maxiter=fit_args["maxiter"],
                popsize=fit_args["popsize"],
                mutation=fit_args["mutation"],
                recombination=fit_args["recombination"],
                tol=fit_args["tol"],
                polish=fit_args["polish"],
                updating=fit_args["updating"],
                workers=fit_args["workers"],
            )

            # pe = results["x"]
            # x_pred = sim_func(pe, args)[1]
            # fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 8))
            # ax[0, 0].plot(x_obs, label='observed')
            # ax[0, 0].plot(x_pred, label='predicted')
            # ax[0, 0].legend()
            # plt.show()

            fout = os.path.join(
                froot,
                "fit_results_" + "_cnd" + str(cnd) + "_sub_" + str(sub) +
                ".txt",
            )
            with open(fout, "w") as f:
                tmp = np.concatenate((results["x"], [results["fun"]]))
                tmp = np.reshape(tmp, (tmp.shape[0], 1))
                np.savetxt(f, tmp.T, "%0.4f", delimiter=",", newline="\n")


def sim_func(params, args):

    alpha_low = params[0]
    alpha_high = params[1]
    beta_low = params[2]
    beta_high = params[3]

    r = args[0]
    x_obs = args[1]
    su = args[2]
    phase = args[3]

    n_trials = r.shape[0]

    delta = np.zeros(n_trials)
    xff = np.zeros(n_trials)
    yff = np.zeros(n_trials)

    for i in range(n_trials - 1):

        yff[i] = xff[i]

        if phase[i] != 3:
            delta[i] = yff[i] + r[i]
        else:
            delta[i] = 0.0

        if su[i] == "low":
            xff[i + 1] = beta_low * xff[i] + alpha_low * delta[i]
        else:
            xff[i + 1] = beta_high * xff[i] + alpha_high * delta[i]

    return (yff, xff)


def obj_func(params, *args):
    obs = args

    rot = obs[0]
    x_obs = obs[1]
    su = obs[2]
    phase = obs[3]

    args = (rot, x_obs, su, phase)

    x_pred = sim_func(params, args)[0]

    sse = np.sum((x_obs - x_pred)**2)

    return sse


def inspect_results_ss(dp):

    froot = "../fits/"

    fits_blocked_high_low = []
    for f in os.listdir(froot):
        if f.startswith("fit_results__cndBlocked - High Low_sub_"):
            d = pd.read_csv(os.path.join(froot, f), header=None)
            d.columns = [
                "alpha_low", "alpha_high", "beta_low", "beta_high", "sse"
            ]
            d["condition"] = "Blocked - High Low"
            d["subject"] = int(f.split("_")[-1].split(".")[0])
            fits_blocked_high_low.append(d)

    fits_blocked_low_high = []
    for f in os.listdir(froot):
        if f.startswith("fit_results__cndBlocked - Low High_sub_"):
            d = pd.read_csv(os.path.join(froot, f), header=None)
            d.columns = [
                "alpha_low", "alpha_high", "beta_low", "beta_high", "sse"
            ]
            d["condition"] = "Blocked - Low High"
            d["subject"] = int(f.split("_")[-1].split(".")[0])
            fits_blocked_low_high.append(d)

    fits_interleaved = []
    for f in os.listdir(froot):
        if f.startswith("fit_results__cndinterleaved_sub_"):
            d = pd.read_csv(os.path.join(froot, f), header=None)
            d.columns = [
                "alpha_low", "alpha_high", "beta_low", "beta_high", "sse"
            ]
            d["condition"] = "interleaved"
            d["subject"] = int(f.split("_")[-1].split(".")[0])
            fits_interleaved.append(d)

    dblh = pd.concat(fits_blocked_low_high)
    dbhl = pd.concat(fits_blocked_high_low)
    di = pd.concat(fits_interleaved)

    dfits = pd.concat([dblh, dbhl, di]).reset_index(drop=True)

    # plot boxplot of alpha_low for the Blocked - Low High vs alpha_high for Blocked - High Low
    dfits["alpha"] = 0.0
    dfits.loc[dfits["condition"] == "Blocked - Low High",
              "alpha"] = dfits.loc[dfits["condition"] == "Blocked - Low High",
                                   "alpha_low"]
    dfits.loc[dfits["condition"] == "Blocked - High Low",
              "alpha"] = dfits.loc[dfits["condition"] == "Blocked - High Low",
                                   "alpha_high"]

    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(8, 8))

    dfits.sort_values(["condition", "subject"], inplace=True)

    alpha_blocked_low = dfits[dfits["condition"] ==
                              "Blocked - Low High"]["alpha_low"]
    alpha_blocked_high = dfits[dfits["condition"] ==
                               "Blocked - High Low"]["alpha_high"]
    beta_blocked_low = dfits[dfits["condition"] ==
                             "Blocked - Low High"]["beta_low"]
    beta_blocked_high = dfits[dfits["condition"] ==
                              "Blocked - High Low"]["beta_high"]

    alpha_interleaved_low = dfits[dfits["condition"] ==
                                  "interleaved"]["alpha_low"]
    alpha_interleaved_high = dfits[dfits["condition"] ==
                                   "interleaved"]["alpha_high"]
    beta_interleaved_low = dfits[dfits["condition"] ==
                                 "interleaved"]["beta_low"]
    beta_interleaved_high = dfits[dfits["condition"] ==
                                  "interleaved"]["beta_high"]

    ax[0, 0].boxplot([alpha_blocked_low, alpha_blocked_high], positions=[0, 1])
    ax[0, 1].boxplot([beta_blocked_low, beta_blocked_high], positions=[0, 1])
    ax[1, 0].boxplot([alpha_interleaved_low, alpha_interleaved_high],
                     positions=[0, 1])
    ax[1, 1].boxplot([beta_interleaved_low, beta_interleaved_high],
                     positions=[0, 1])

    ax[0, 0].scatter(np.zeros_like(alpha_blocked_low),
                     alpha_blocked_low,
                     color="k",
                     alpha=0.5)
    ax[0, 0].scatter(np.ones_like(alpha_blocked_high),
                     alpha_blocked_high,
                     color="k",
                     alpha=0.5)
    ax[0, 1].scatter(np.zeros_like(beta_blocked_low),
                     beta_blocked_low,
                     color="k",
                     alpha=0.5)
    ax[0, 1].scatter(np.ones_like(beta_blocked_high),
                     beta_blocked_high,
                     color="k",
                     alpha=0.5)
    ax[1, 0].scatter(np.zeros_like(alpha_interleaved_low),
                     alpha_interleaved_low,
                     color="k",
                     alpha=0.5)
    ax[1, 0].scatter(np.ones_like(alpha_interleaved_high),
                     alpha_interleaved_high,
                     color="k",
                     alpha=0.5)
    ax[1, 1].scatter(np.zeros_like(beta_interleaved_low),
                     beta_interleaved_low,
                     color="k",
                     alpha=0.5)
    ax[1, 1].scatter(np.ones_like(beta_interleaved_high),
                     beta_interleaved_high,
                     color="k",
                     alpha=0.5)

    for low, high in zip(alpha_interleaved_low, alpha_interleaved_high):
        ax[1, 0].plot([0, 1], [low, high], color="k", alpha=0.3)

    for low, high in zip(beta_interleaved_low, beta_interleaved_high):
        ax[1, 1].plot([0, 1], [low, high], color="k", alpha=0.3)

    ax[0, 0].set_xticklabels(["Low", "High"])
    ax[0, 1].set_xticklabels(["Low", "High"])
    ax[1, 0].set_xticklabels(["Low", "High"])
    ax[1, 1].set_xticklabels(["Low", "High"])

    ax[0, 0].set_ylabel("Alpha")
    ax[0, 1].set_ylabel("Beta")
    ax[1, 0].set_ylabel("Alpha")
    ax[1, 1].set_ylabel("Beta")

    ax[0, 0].set_title("Blocked Condition")
    ax[0, 1].set_title("Blocked Condition")
    ax[1, 0].set_title("Interleaved Condition")
    ax[1, 1].set_title("Interleaved Condition")

    plt.savefig("../figures/ssm_fit_parameters.png")
    plt.close()

    print()
    print("alpha blocked low vs high")
    print(pg.ttest(alpha_blocked_low, alpha_blocked_high, paired=False))
    print()
    print("beta blocked low vs high")
    print(pg.ttest(beta_blocked_low, beta_blocked_high, paired=False))
    print()
    print("alpha interleaved low vs high")
    print(pg.ttest(alpha_interleaved_low, alpha_interleaved_high, paired=True))
    print()
    print("beta interleaved low vs high")
    print(pg.ttest(beta_interleaved_low, beta_interleaved_high, paired=True))

    cnds = dfits["condition"].unique()
    d_pred_list = []
    for j, c in enumerate(cnds):
        dc = dp[dp["condition"] == c]
        for s in dc["subject"].unique():
            ds = dc[dc["subject"] == s]
            ds = ds[ds["phase"].isin([2, 3])]
            ds = ds.iloc[:225, :].reset_index(drop=True)
            fs = dfits[(dfits["subject"] == s) & (dfits["condition"] == c)]
            alpha_low = fs["alpha_low"].values[0]
            alpha_high = fs["alpha_high"].values[0]
            beta_low = fs["beta_low"].values[0]
            beta_high = fs["beta_high"].values[0]

            su = ds["su"].values
            phase = ds["phase"].values
            r = ds["rotation"].values
            x_obs = ds["emv"].values

            params = (alpha_low, alpha_high, beta_low, beta_high)
            args = (r, x_obs, su, phase)

            d_pred = sim_func(params, args)[1]
            d_pred = pd.DataFrame(d_pred, columns=["emv"])
            d_pred["trial"] = np.arange(d_pred.shape[0])
            d_pred["subject"] = s
            d_pred["condition"] = c
            d_pred_list.append(d_pred)

    d_pred_all = pd.concat(d_pred_list).reset_index(drop=True)
    fig, ax = plt.subplots(1,
                           len(cnds),
                           squeeze=False,
                           figsize=(8 * len(cnds), 6))
    for j, c in enumerate(cnds):
        dc = d_pred_all[d_pred_all["condition"] == c]
        for s in dc["subject"].unique():
            ds = dc[dc["subject"] == s]
            ax[0, j].plot(ds["trial"], ds["emv"], alpha=0.5)
        ax[0, j].set_title(c)
        ax[0, j].set_xlabel("Trial")
        ax[0, j].set_ylabel("Predicted EMV")

    plt.savefig("../figures/ssm_fit_predictions.png")
    plt.close()


def interpolate_movements(d):
    t = d["t"]
    x = d["x"]
    y = d["y"]
    v = d["v"]

    xs = CubicSpline(t, x)
    ys = CubicSpline(t, y)
    vs = CubicSpline(t, v)

    tt = np.linspace(t.min(), t.max(), 100)
    xx = xs(tt)
    yy = ys(tt)
    vv = vs(tt)

    relsamp = np.arange(0, tt.shape[0], 1)

    dd = pd.DataFrame({"relsamp": relsamp, "t": tt, "x": xx, "y": yy, "v": vv})
    dd["condition"] = d["condition"].unique()[0]
    dd["subject"] = d["subject"].unique()[0]
    dd["trial"] = d["trial"].unique()[0]
    dd["phase"] = d["phase"].unique()[0]
    dd["su"] = d["su"].unique()[0]
    dd["imv"] = d["imv"].unique()[0]
    dd["emv"] = d["emv"].unique()[0]

    return dd


def compute_kinematics(d):
    t = d["t"].to_numpy()
    x = d["x"].to_numpy()
    y = d["y"].to_numpy()

    x = x - x[0]
    y = y - y[0]
    y = -y

    r = np.sqrt(x**2 + y**2)
    theta = (np.arctan2(y, x)) * 180 / np.pi

    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    v = np.sqrt(vx**2 + vy**2)

    v_peak = v.max()
    # ts = t[v > (0.05 * v_peak)][0]
    ts = t[r > 0.1 * r.max()][0]

    imv = theta[(t >= ts) & (t <= ts + 0.1)].mean()
    emv = theta[-1]

    d["x"] = x
    d["y"] = y
    d["v"] = v
    d["imv"] = 90 - imv
    d["emv"] = 90 - emv

    return d


def load_data():

    dir_data = "../data/"

    d_rec = []

    # iterate over files in the ../data directory
    for f in os.listdir(dir_data):

        if f.endswith(".csv"):

            # extract subject number
            s = int(f.split("_")[1])

            # try to load both trial and movement files
            try:
                f_trl = "sub_{}_data.csv".format(s)
                f_mv = "sub_{}_data_move.csv".format(s)

                d_trl = pd.read_csv(os.path.join(dir_data, f_trl))
                d_mv = pd.read_csv(os.path.join(dir_data, f_mv))

                if d_trl.shape[0] != 429:
                    print("Subject {} has anomolous trial data".format(s))

                else:
                    d_trl = d_trl.sort_values(
                        ["condition", "subject", "trial"])
                    d_mv = d_mv.sort_values(
                        ["condition", "subject", "t", "trial"])

                    d_hold = d_mv[d_mv["state"].isin(["state_holding"])]
                    x_start = d_hold.x.mean()
                    y_start = d_hold.y.mean()

                    d_mv = d_mv[d_mv["state"].isin(["state_moving"])]

                    phase = np.zeros(d_trl["trial"].nunique())
                    phase[:30] = 1
                    phase[30:130] = 2
                    phase[130:180] = 3
                    phase[180:230] = 4
                    phase[230:330] = 5
                    phase[330:380] = 6
                    phase[380:] = 7
                    d_trl["phase"] = phase

                    d_trl["su"] = d_trl["su"].astype("category")
                    d_trl["ep"] = (d_trl["ep"] * 180 / np.pi) + 90
                    d_trl["rotation"] = d_trl["rotation"] * 180 / np.pi

                    d = pd.merge(d_mv,
                                 d_trl,
                                 how="outer",
                                 on=["condition", "subject", "trial"])

                    d = d.groupby(["condition", "subject", "trial"],
                                  group_keys=False).apply(compute_kinematics)

                    d_rec.append(d)

            # print warning if file load fails
            except Exception as e:
                print("Could not load data for subject {}: {}".format(s, e))

    d = pd.concat(d_rec)

    d["su"] = d["su"].cat.rename_categories({0.0: "low", 26.78: "high"})

    d.groupby(["condition"])["subject"].unique()
    d.groupby(["condition"])["subject"].nunique()

    d.sort_values(["condition", "subject", "trial", "t"], inplace=True)

    for s in d["subject"].unique():
        ds = d[d["subject"] == s]
        if ds["condition"].unique() == "blocked":
            if ds[ds["phase"] == 2]["su"].unique() == "low":
                d.loc[d["subject"] == s, "condition"] = "Blocked - Low High"
            else:
                d.loc[d["subject"] == s, "condition"] = "Blocked - High Low"

    d.groupby(["condition"])["subject"].unique()
    d.groupby(["condition"])["subject"].nunique()
    d.groupby(["condition", "subject"])["trial"].nunique()

    # NOTE: create by trial frame
    dp = d[["condition", "subject", "trial", "phase", "su", "emv",
            "rotation"]].drop_duplicates()

    dp["emv"] = -dp["emv"]

    def identify_outliers_mad(x, col="emv", thresh=3.5):
        """
        Robust outlier detection using modified z-scores (MAD).
        thresh=3.5 is the standard recommended cutoff (Iglewicz & Hoaglin).
        """
        x = x.copy()
        x["outlier"] = False

        median = x[col].median()
        mad = np.median(np.abs(x[col] - median))

        if mad == 0 or np.isnan(mad):
            return x

        modified_z = 0.6745 * (x[col] - median) / mad
        x.loc[np.abs(modified_z) > thresh, "outlier"] = True

        return x

    dp = dp.groupby(["condition", "subject", "phase"
                     ]).apply(identify_outliers_mad).reset_index(drop=True)

    #    # iterate over subjects and plot emv per trial with outliers marked
    #    for s in dp["subject"].unique():
    #        ds = dp[dp["subject"] == s]
    #        fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 6))
    #        sns.lineplot(
    #            data=ds[ds["outlier"] == False],
    #            x="trial",
    #            y="emv",
    #            hue="outlier",
    #            legend="full",
    #            ax=ax[0, 0],
    #        )
    #        sns.scatterplot(
    #            data=ds,
    #            x="trial",
    #            y="emv",
    #            hue="outlier",
    #            style="phase",
    #            legend=None,
    #            ax=ax[0, 0],
    #        )
    #        plt.show()

    dp.groupby(["condition", "subject"])["outlier"].sum()
    dp = dp[dp["outlier"] == False]
    dp = dp.sort_values(["condition", "subject", "trial"])

    # baseline correct per subject and condition
    dp["emv"] = dp["emv"] - dp.groupby([
        "subject", "condition"
    ])["emv"].transform(lambda x: x[dp["phase"] == 1].mean())

    def add_prev(x):
        x["su_prev"] = x["su"].shift(1)
        x["delta_emv"] = x["emv"].diff()
        x["movement_error"] = -x["rotation"] + x["emv"]
        x["movement_error_prev"] = x["movement_error"].shift(1)
        return x

    dp = dp.groupby(["condition", "subject"], group_keys=False).apply(add_prev)

    return dp


def make_fig_per_subject(dp):

    # NOTE: inspect individual subjects --- measures
    for i, s in enumerate(dp["subject"].unique()):

        ds = dp[dp["subject"] == s].copy()

        fig, ax = plt.subplots(3, 1, squeeze=False, figsize=(5, 12))
        fig.subplots_adjust(wspace=0.3, hspace=0.5)

        sns.scatterplot(
            data=ds,
            x="trial",
            y="emv",
            hue="su_prev",
            markers=True,
            legend="full",
            ax=ax[0, 0],
        )
        sns.scatterplot(
            data=ds,
            x="trial",
            y="movement_error",
            hue="su_prev",
            markers=True,
            legend=False,
            ax=ax[1, 0],
        )
        sns.scatterplot(
            data=ds,
            x="trial",
            y="delta_emv",
            hue="su_prev",
            markers=True,
            legend=False,
            ax=ax[2, 0],
        )
        [
            sns.lineplot(
                data=ds,
                x="trial",
                y="rotation",
                hue="condition",
                palette=['k'],
                legend=False,
                ax=ax_,
            ) for ax_ in [ax[0, 0], ax[1, 0], ax[2, 0]]
        ]

        ax[0, 0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.4), ncol=2)

        plt.savefig("../figures/fig_measures_sub_" + str(s) + ".png")
        plt.close()


def make_fig_group_per_trial(dpp):

    fig, ax = plt.subplots(3, 1, squeeze=False, figsize=(8, 12))
    fig.subplots_adjust(wspace=0.3, hspace=0.3, top=0.95, bottom=0.05)

    sns.scatterplot(
        data=dpp[dpp["condition"] == "Blocked - High Low"],
        x="trial",
        y="emv",
        style="phase",
        hue="su_prev",
        markers=True,
        legend="full",
        ax=ax[0, 0],
    )
    sns.scatterplot(
        data=dpp[dpp["condition"] == "Blocked - Low High"],
        x="trial",
        y="emv",
        style="phase",
        hue="su_prev",
        markers=True,
        legend="full",
        ax=ax[1, 0],
    )
    sns.scatterplot(
        data=dpp[dpp["condition"] == "interleaved"],
        x="trial",
        y="emv",
        style="phase",
        hue="su_prev",
        markers=True,
        legend="full",
        ax=ax[2, 0],
    )
    [x.set_ylim(-10, 40) for x in [ax[0, 0], ax[1, 0], ax[2, 0]]]
    [x.set_xlabel("Trial") for x in [ax[0, 0], ax[1, 0], ax[2, 0]]]
    [
        x.set_ylabel("Endppoint Movement Vector")
        for x in [ax[0, 0], ax[1, 0], ax[2, 0]]
    ]
    [
        sns.lineplot(
            data=dpp[dpp["condition"] != "interleaved"],
            x="trial",
            y="rotation",
            hue="condition",
            palette=['k'],
            legend=False,
            ax=ax_,
        ) for ax_ in [ax[0, 0], ax[1, 0], ax[2, 0]]
    ]
    ax[0, 0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=4)
    ax[1, 0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=4)
    ax[2, 0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=4)

    plt.savefig("../figures/summary_per_trial.png")
    plt.close()


def fit_regression(d):

    d["exp_fast"] = 1 - np.exp(-0.3 * d["trial"])
    d["exp_med"] = 1 - np.exp(-0.03 * d["trial"])
    md = smf.mixedlm(
        "emv ~ C(su_prev, Diff)*movement_error_prev + exp_med + exp_fast",
        data=d,
        groups=d["subject"])

    mdf = md.fit()
    print(mdf.summary())

    d["emv_pred"] = mdf.model.predict(mdf.params, mdf.model.exog)

    # plot obs and pred overliad
    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(6, 6))
    fig.subplots_adjust(wspace=0.3, hspace=0.5)
    sns.lineplot(data=d,
                 x="trial",
                 y="emv",
                 hue="su_prev",
                 markers=True,
                 ax=ax[0, 0])
    sns.lineplot(data=d,
                 x="trial",
                 y="emv_pred",
                 hue="su_prev",
                 markers=True,
                 ax=ax[0, 1])
    [x.set_ylim(-5, 20) for x in [ax[0, 0], ax[0, 1]]]
    plt.show()
