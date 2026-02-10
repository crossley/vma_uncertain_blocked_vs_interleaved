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

    for sim_func in [
            sim_func_single_su, sim_func_dual_su_alpha_only,
            sim_func_dual_su_beta_only, sim_func_dual_su
    ]:

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

                args = (rot, x_obs, su, phase, fit_args["sim_func"])

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
                    "fit_results_" + str(sim_func.__name__) + "_cnd" +
                    str(cnd) + "_sub_" + str(sub) + ".txt",
                )
                with open(fout, "w") as f:
                    tmp = np.concatenate((results["x"], [results["fun"]]))
                    tmp = np.reshape(tmp, (tmp.shape[0], 1))
                    np.savetxt(f, tmp.T, "%0.4f", delimiter=",", newline="\n")


def sim_func_dual_su(params, args):

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
            delta[i] = -yff[i] + r[i]
        else:
            delta[i] = 0.0

        if su[i] == "low":
            xff[i + 1] = beta_low * xff[i] + alpha_low * delta[i]
        else:
            xff[i + 1] = beta_high * xff[i] + alpha_high * delta[i]

    return (yff, xff)


def sim_func_dual_su_alpha_only(params, args):

    alpha_low = params[0]
    alpha_high = params[1]
    beta = params[2]

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
            delta[i] = -yff[i] + r[i]
        else:
            delta[i] = 0.0

        if su[i] == "low":
            xff[i + 1] = beta * xff[i] + alpha_low * delta[i]
        else:
            xff[i + 1] = beta * xff[i] + alpha_high * delta[i]

    return (yff, xff)


def sim_func_dual_su_beta_only(params, args):

    alpha = params[0]
    beta_low = params[1]
    beta_high = params[2]

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
            delta[i] = -yff[i] + r[i]
        else:
            delta[i] = 0.0

        if su[i] == "low":
            xff[i + 1] = beta_low * xff[i] + alpha * delta[i]
        else:
            xff[i + 1] = beta_high * xff[i] + alpha * delta[i]

    return (yff, xff)


def sim_func_single_su(params, args):

    alpha = params[0]
    beta = params[1]
    gamma = params[2]

    r = args[0]
    x_obs = args[1]
    su = args[2]
    phase = args[3]

    n_trials = r.shape[0]

    delta = np.zeros(n_trials)
    xff = np.zeros(n_trials)
    yff = np.zeros(n_trials)

    su_err_scale = 0.5

    for i in range(n_trials - 1):

        yff[i] = xff[i]

        if phase[i] != 3:
            delta[i] = -yff[i] + r[i]
        else:
            delta[i] = 0.0

        if su[i] == "low":
            suv = 0.9
        elif su[i] == "high":
            suv = 0.1

        su_err_scale = su_err_scale + gamma * (suv - su_err_scale)

        xff[i + 1] = beta * xff[i] + su_err_scale * alpha * delta[i]

    return (yff, xff)


def obj_func(params, *args):

    obs = args

    rot = obs[0]
    x_obs = obs[1]
    su = obs[2]
    phase = obs[3]
    sim_func = obs[4]

    args = (rot, x_obs, su, phase)

    x_pred = sim_func(params, args)[0]

    sse = np.sum((x_obs - x_pred)**2)

    return sse


def psp_ss():

    bin_edges = np.arange(-20, 11, 1)
    n = 200

    r = np.concatenate([np.zeros(n // 2), np.ones(n // 2) * 15])
    x_obs = np.zeros(n)
    phase = np.array([2] * n)

    psp_record = []

    def get_slope(sim_func, params, su, bin_edges, mask_su=None):

        su_prev = np.roll(su, 1)
        args = (r, x_obs, su, phase)

        x_pred_emv = sim_func(params, args)[1]
        x_pred_delta_emv = np.diff(x_pred_emv, prepend=0)
        x_pred_err = -r + x_pred_emv
        x_pred_err_prev = np.roll(x_pred_err, 1)

        df = pd.DataFrame({
            "trial": np.arange(n),
            "emv": x_pred_emv,
            "delta_emv": x_pred_delta_emv,
            "movement_error": x_pred_err,
            "movement_error_prev": x_pred_err_prev,
            "su_prev": su_prev,
        })

        df["me_prev_bin"] = pd.cut(
            df["movement_error_prev"],
            bins=bin_edges,
            labels=False,
        )

        if mask_su is not None:
            df = df.loc[df["su_prev"] == mask_su]

        df = df[df["trial"] > 100].reset_index(drop=True)

        X = df[["movement_error_prev"]]
        y = df["delta_emv"]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()

        return model.params["movement_error_prev"]

    def psp_sim_func(sim_func,
                     param_grid,
                     param_keys,
                     model_name,
                     epsilon=1e-2):

        psp_record = []

        for param_dict in param_grid:
            # build the params tuple in the order expected by sim_func
            params = tuple(param_dict[k] for k in param_keys)

            print(model_name, param_dict)

            # NOTE: Blocked - Low
            su_blocked_low = np.array(["low"] * n)
            slope_blocked_low = get_slope(
                sim_func=sim_func,
                params=params,
                su=su_blocked_low,
                bin_edges=bin_edges,
            )

            # NOTE: Blocked - High
            su_blocked_high = np.array(["high"] * n)
            slope_blocked_high = get_slope(
                sim_func=sim_func,
                params=params,
                su=su_blocked_high,
                bin_edges=bin_edges,
            )

            # NOTE: Interleaved
            su_interleaved = np.array(["high"] * (n // 2) + ["low"] * (n // 2))
            np.random.shuffle(su_interleaved)

            slope_interleaved_low = get_slope(
                sim_func=sim_func,
                params=params,
                su=su_interleaved,
                bin_edges=bin_edges,
                mask_su="low",
            )

            slope_interleaved_high = get_slope(
                sim_func=sim_func,
                params=params,
                su=su_interleaved,
                bin_edges=bin_edges,
                mask_su="high",
            )

            row = {
                "model": model_name,
                "slope_blocked_low": slope_blocked_low,
                "slope_blocked_high": slope_blocked_high,
                "slope_interleaved_low": slope_interleaved_low,
                "slope_interleaved_high": slope_interleaved_high,
            }
            # add all params into the row too
            row.update(param_dict)

            psp_record.append(row)

        psp_df = pd.DataFrame(psp_record)

        # magnitudes + diffs
        psp_df["slope_blocked_low"] = psp_df["slope_blocked_low"].abs()
        psp_df["slope_blocked_high"] = psp_df["slope_blocked_high"].abs()
        psp_df["slope_interleaved_low"] = psp_df["slope_interleaved_low"].abs()
        psp_df["slope_interleaved_high"] = psp_df[
            "slope_interleaved_high"].abs()

        psp_df["slope_blocked_diff"] = (psp_df["slope_blocked_low"] -
                                        psp_df["slope_blocked_high"])
        psp_df["slope_interleaved_diff"] = (psp_df["slope_interleaved_low"] -
                                            psp_df["slope_interleaved_high"])

        psp_df["psp_class"] = "None"
        psp_df.loc[(psp_df["slope_blocked_diff"] > epsilon),
                   "psp_class"] = "Block Satisfied"
        psp_df.loc[(psp_df["slope_interleaved_diff"] > epsilon),
                   "psp_class"] = "Interleaved Satisfied"
        psp_df.loc[(psp_df["slope_blocked_diff"] > epsilon) &
                   (psp_df["slope_interleaved_diff"] > epsilon),
                   "psp_class"] = "Both Satisfied"

        return psp_df

    # single-rate model ranges (your current example)
    alpha_range = np.arange(0, 1, 0.1)
    beta_range = np.arange(0, 1, 0.1)
    gamma_range = np.arange(0, 1, 0.1)

    # dual_su_alpha_only
    alpha_low_range = np.arange(0, 1, 0.1)
    alpha_high_range = np.arange(0, 1, 0.1)
    beta_range_alpha_only = np.arange(0, 1, 0.1)

    # dual_su_beta_only
    alpha_range_beta_only = np.arange(0, 1, 0.1)
    beta_low_range = np.arange(0, 1, 0.1)
    beta_high_range = np.arange(0, 1, 0.1)

    # dual_su (full)
    alpha_low_range_full = np.arange(0, 1, 0.1)
    alpha_high_range_full = np.arange(0, 1, 0.1)
    beta_low_range_full = np.arange(0, 1, 0.1)
    beta_high_range_full = np.arange(0, 1, 0.1)

    single_param_grid = [{
        "alpha": a,
        "beta": b,
        "gamma": g
    } for a in alpha_range for b in beta_range for g in gamma_range]
    single_param_keys = ["alpha", "beta", "gamma"]

    alpha_only_param_grid = [{
        "alpha_low": al,
        "alpha_high": ah,
        "beta": b
    } for al in alpha_low_range for ah in alpha_high_range
                             for b in beta_range_alpha_only]
    alpha_only_param_keys = ["alpha_low", "alpha_high", "beta"]

    beta_only_param_grid = [{
        "alpha": a,
        "beta_low": bl,
        "beta_high": bh
    } for a in alpha_range_beta_only for bl in beta_low_range
                            for bh in beta_high_range]
    beta_only_param_keys = ["alpha", "beta_low", "beta_high"]

    dual_full_param_grid = [{
        "alpha_low": al,
        "alpha_high": ah,
        "beta_low": bl,
        "beta_high": bh
    } for al in alpha_low_range_full for ah in alpha_high_range_full
                            for bl in beta_low_range_full
                            for bh in beta_high_range_full]
    dual_full_param_keys = ["alpha_low", "alpha_high", "beta_low", "beta_high"]

    psp_single = psp_sim_func(
        sim_func=sim_func_single_su,
        param_grid=single_param_grid,
        param_keys=single_param_keys,
        model_name="single_su",
    )

    psp_dual_alpha_only = psp_sim_func(
        sim_func=sim_func_dual_su_alpha_only,
        param_grid=alpha_only_param_grid,
        param_keys=alpha_only_param_keys,
        model_name="dual_su_alpha_only",
    )

    psp_dual_beta_only = psp_sim_func(
        sim_func=sim_func_dual_su_beta_only,
        param_grid=beta_only_param_grid,
        param_keys=beta_only_param_keys,
        model_name="dual_su_beta_only",
    )

    psp_dual_full = psp_sim_func(
        sim_func=sim_func_dual_su,
        param_grid=dual_full_param_grid,
        param_keys=dual_full_param_keys,
        model_name="dual_su",
    )

    # order psp_class in all dataframes
    psp_single["psp_class"] = pd.Categorical(
        psp_single["psp_class"],
        categories=[
            "None", "Block Satisfied", "Interleaved Satisfied", "Both Satisfied"
        ],
        ordered=True,
    )

    psp_dual_alpha_only["psp_class"] = pd.Categorical(
        psp_dual_alpha_only["psp_class"],
        categories=[
            "None", "Block Satisfied", "Interleaved Satisfied", "Both Satisfied"
        ],
        ordered=True,
    )

    psp_dual_beta_only["psp_class"] = pd.Categorical(
        psp_dual_beta_only["psp_class"],
        categories=[
            "None", "Block Satisfied", "Interleaved Satisfied", "Both Satisfied"
        ],
        ordered=True,
    )

    psp_dual_full["psp_class"] = pd.Categorical(
        psp_dual_full["psp_class"],
        categories=[
            "None", "Block Satisfied", "Interleaved Satisfied", "Both Satisfied"
        ],
        ordered=True,
    )

    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(12, 6))
    sns.scatterplot(data=psp_single,
                    x="slope_blocked_low",
                    y="slope_blocked_high",
                    hue="psp_class",
                    ax=ax[0, 0])
    sns.scatterplot(data=psp_single,
                    x="slope_interleaved_low",
                    y="slope_interleaved_high",
                    hue="psp_class",
                    ax=ax[0, 1])
    plt.tight_layout()
    plt.savefig("../figures/psp_single_su.png")

    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(12, 6))
    sns.scatterplot(data=psp_dual_alpha_only,
                    x="slope_blocked_low",
                    y="slope_blocked_high",
                    hue="psp_class",
                    ax=ax[0, 0])
    sns.scatterplot(data=psp_dual_alpha_only,
                    x="slope_interleaved_low",
                    y="slope_interleaved_high",
                    hue="psp_class",
                    ax=ax[0, 1])
    plt.tight_layout()
    plt.savefig("../figures/psp_dual_alpha_only_su.png")

    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(12, 6))
    sns.scatterplot(data=psp_dual_beta_only,
                    x="slope_blocked_low",
                    y="slope_blocked_high",
                    hue="psp_class",
                    ax=ax[0, 0])
    sns.scatterplot(data=psp_dual_beta_only,
                    x="slope_interleaved_low",
                    y="slope_interleaved_high",
                    hue="psp_class",
                    ax=ax[0, 1])
    plt.tight_layout()
    plt.savefig("../figures/psp_dual_beta_only_su.png")

    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(12, 6))
    sns.scatterplot(data=psp_dual_full,
                    x="slope_blocked_low",
                    y="slope_blocked_high",
                    hue="psp_class",
                    ax=ax[0, 0])
    sns.scatterplot(data=psp_dual_full,
                    x="slope_interleaved_low",
                    y="slope_interleaved_high",
                    hue="psp_class",
                    ax=ax[0, 1])
    plt.tight_layout()
    plt.savefig("../figures/psp_dual_full_su.png")

    # plot parameters for both class
    from pandas.plotting import parallel_coordinates

    psp_single_both = psp_single[psp_single["psp_class"] == "Both Satisfied"].copy()
    if not psp_single_both.empty:
        non_param_cols = {
            "psp_class",
            "slope_blocked_low",
            "slope_blocked_high",
            "slope_interleaved_low",
            "slope_interleaved_high",
            "slope_blocked_diff",
            "slope_interleaved_diff",
            "model",  # safe even if not present
        }
        param_cols = [c for c in psp_single_both.columns if c not in non_param_cols]

        df_plot = psp_single_both[param_cols].copy()
        df_plot["cls"] = "both"

        plt.figure(figsize=(8, 4))
        parallel_coordinates(df_plot, class_column="cls", alpha=0.5)
        plt.title("PSP parameters (Both Satisfied) – single_su")
        plt.xlabel("Parameters")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig("../figures/psp_single_su_parameters_both.png")

    # --- dual_su_alpha_only ---
    psp_dual_alpha_only_both = psp_dual_alpha_only[
        psp_dual_alpha_only["psp_class"] == "Both Satisfied"
    ].copy()

    if not psp_dual_alpha_only_both.empty:
        non_param_cols = {
            "psp_class",
            "slope_blocked_low",
            "slope_blocked_high",
            "slope_interleaved_low",
            "slope_interleaved_high",
            "slope_blocked_diff",
            "slope_interleaved_diff",
            "model",
        }
        param_cols = [c for c in psp_dual_alpha_only_both.columns if c not in non_param_cols]

        df_plot = psp_dual_alpha_only_both[param_cols].copy()
        df_plot["cls"] = "both"

        plt.figure(figsize=(8, 4))
        parallel_coordinates(df_plot, class_column="cls", alpha=0.5)
        plt.title("PSP parameters (Both Satisfied) – dual_su_alpha_only")
        plt.xlabel("Parameters")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig("../figures/psp_dual_alpha_only_parameters_both.png")

    # --- dual_su_beta_only ---
    psp_dual_beta_only_both = psp_dual_beta_only[
        psp_dual_beta_only["psp_class"] == "Both Satisfied"
    ].copy()

    if not psp_dual_beta_only_both.empty:
        non_param_cols = {
            "psp_class",
            "slope_blocked_low",
            "slope_blocked_high",
            "slope_interleaved_low",
            "slope_interleaved_high",
            "slope_blocked_diff",
            "slope_interleaved_diff",
            "model",
        }
        param_cols = [c for c in psp_dual_beta_only_both.columns if c not in non_param_cols]

        df_plot = psp_dual_beta_only_both[param_cols].copy()
        df_plot["cls"] = "both"

        plt.figure(figsize=(8, 4))
        parallel_coordinates(df_plot, class_column="cls", alpha=0.5)
        plt.title("PSP parameters (Both Satisfied) – dual_su_beta_only")
        plt.xlabel("Parameters")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig("../figures/psp_dual_beta_only_parameters_both.png")

    # --- dual_su (full) ---
    psp_dual_full_both = psp_dual_full[
        psp_dual_full["psp_class"] == "Both Satisfied"
    ].copy()

    if not psp_dual_full_both.empty:
        non_param_cols = {
            "psp_class",
            "slope_blocked_low",
            "slope_blocked_high",
            "slope_interleaved_low",
            "slope_interleaved_high",
            "slope_blocked_diff",
            "slope_interleaved_diff",
            "model",
        }
        param_cols = [c for c in psp_dual_full_both.columns if c not in non_param_cols]

        df_plot = psp_dual_full_both[param_cols].copy()
        df_plot["cls"] = "both"

        plt.figure(figsize=(8, 4))
        parallel_coordinates(df_plot, class_column="cls", alpha=0.5)
        plt.title("PSP parameters (Both Satisfied) – dual_su")
        plt.xlabel("Parameters")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig("../figures/psp_dual_full_parameters_both.png")

    # fixed class order so bars line up visually across models
    class_order = ["None", "Block Satisfied", "Interleaved Satisfied", "Both Satisfied"]

    # collect models in a dict: name -> df
    psp_models = {
        "single_su": psp_single,
        "dual_su_alpha_only": psp_dual_alpha_only,
        "dual_su_beta_only": psp_dual_beta_only,
        "dual_su_full": psp_dual_full,
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)

    for ax, (model_name, df) in zip(axes.flat, psp_models.items()):
        # proportion of parameter space in each class
        prop = (
            df["psp_class"]
            .value_counts(normalize=True)
            .reindex(class_order, fill_value=0.0)
        )

        sns.barplot(
            x=prop.index,
            y=prop.values,
            ax=ax,
        )
        ax.set_title(model_name)
        ax.set_xlabel("PSP class")
        ax.set_ylabel("Proportion of parameter space")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig("../figures/psp_class_proportions_all_models.png")

    # fixed class order so bars line up visually across models
    class_order = ["None", "Block Satisfied", "Interleaved Satisfied", "Both Satisfied"]

    # collect models in a dict: name -> df
    psp_models = {
        "single_su": psp_single,
        "dual_su_alpha_only": psp_dual_alpha_only,
        "dual_su_beta_only": psp_dual_beta_only,
        "dual_su_full": psp_dual_full,
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)

    for ax, (model_name, df) in zip(axes.flat, psp_models.items()):
        # proportion of parameter space in each class
        prop = (
            df["psp_class"]
            .value_counts(normalize=True)
            .reindex(class_order, fill_value=0.0)
        )

        sns.barplot(
            x=prop.index,
            y=prop.values,
            ax=ax,
        )
        ax.set_title(model_name)
        ax.set_xlabel("PSP class")
        ax.set_ylabel("Proportion of parameter space")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig("../figures/psp_class_proportions_all_models.png")


def inspect_results_ss_model_comparison(dp):

    froot = "../fits/"
    fit_result_rec = []
    for fname in os.listdir(froot):
        if fname.endswith(".txt"):
            path = os.path.join(froot, fname)
            fit_result = pd.read_csv(path,
                                     header=None,
                                     names=["p1", "p2", "p3", "p4", "sse"])

            # TODO: get correct n
            n = 100
            if "single_su" in fname:
                model = "single_su"
                k = 3
                print("made it")
            if "dual_su_alpha_only" in fname:
                model = "dual_su_alpha_only"
                k = 3
            if "dual_su_beta_only" in fname:
                model = "dual_su_beta_only"
                k = 3
            if "dual_su_cnd" in fname:
                model = "dual_su"
                k = 4

            if "interleaved" in fname:
                condition = "interleaved"
            elif "Blocked - High" in fname:
                condition = "blocked_high"
            elif "Blocked - Low" in fname:
                condition = "blocked_low"
            else:
                condition = None

            subject = int(fname.split("sub_")[-1].split(".")[0])

            fit_result["model"] = model
            fit_result["condition"] = condition
            fit_result["subject"] = subject
            fit_result["bic"] = n * np.log(
                fit_result["sse"] / n) + k * np.log(n)

            fit_result_rec.append(fit_result)

    df = pd.concat(fit_result_rec).reset_index(drop=True)

    # make bar plot of BIC values per model and condition
    sns.set_palette("rocket", 4)
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 6))
    sns.barplot(data=df, x="condition", y="bic", hue="model", ax=ax[0, 0])
    plt.show()


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


def make_fig_dela_emv_vs_movement_error_prev(dp):

    dpp = dp.copy()

    dpp = dpp[dpp["phase"].isin([2])]

    # replace condition names for plotting with "Blocked" and "Interleaved"
    dpp["condition"] = dpp["condition"].replace({
        "Blocked - High Low": "Blocked",
        "Blocked - Low High": "Blocked",
        "interleaved": "Interleaved"
    }).copy()

    # group movement_error_prev into bins and plot delta_emv vs movement_error_prev
    bin_edges = np.arange(-20, 11, 3)
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
