from dataclasses import dataclass

import awkward as ak
import numpy as np
import pandas as pd
import vector
from tqdm.auto import tqdm


DEFAULT_STRING_CODES = [
    "2to4lep",
    "Zmumu",
    "m10_40_Zmumu",
    "Zee",
    "m10_40_Zee",
    "Ztautau",
    "ttbar",
    "Wmunu",
    "Wenu",
    "Wtaunu",
]

DEFAULT_READ_VARIABLES = [
    "lep_n",
    "lep_pt",
    "lep_eta",
    "lep_phi",
    "lep_e",
    "lep_ptvarcone30",
    "lep_topoetcone20",
    "lep_type",
    "lep_charge",
    "trigE",
    "trigM",
    "lep_isTrigMatched",
    "lep_isLooseID",
    "lep_isMediumID",
    "lep_isLooseIso",
    "lep_isTightIso",
]

SCAN_VARS = [
    "lep_n",
    "lep_pt",
    "lep_ptvarcone30",
    "lep_topoetcone20",
    "lep_type",
    "lep_charge",
    "trigE",
    "trigM",
    "mass",
    "charge_product",
    "totalWeight",
]

MC_PREFIXES = [
    "Zee",
    "Zmumu",
    "m10_40_Zee",
    "m10_40_Zmumu",
    "Ztautau",
    "ttbar",
    "Wenu",
    "Wmunu",
    "Wtaunu",
]

COLOR_CYCLE = [
    "k",
    "b",
    "g",
    "r",
    "m",
    "c",
    "y",
    "tab:orange",
    "tab:purple",
    "tab:brown",
]


@dataclass(frozen=True)
class ZllConfig:
    channel: str
    os_dict: dict
    ss_dict: dict
    df_os_dict: dict
    signal_key: str
    produced_signal: float
    nominal_ptcut: int
    nominal_etcut: int
    nominal_mass_window: tuple = (66, 116)
    nominal_mode: str = "df"
    nominal_order: str = "iso -> df -> ss"


def make_preselection_cut(sign="OS", loose_pt=15):
    want_os = sign.upper() == "OS"
    want_ss = sign.upper() == "SS"

    def preselection_cut(data):
        data = data[data["lep_n"] == 2]
        data = data[(data["trigM"]) | (data["trigE"])]
        data = data[(data["lep_isMediumID"][:, 0]) & (data["lep_isMediumID"][:, 1])]
        data = data[(data["lep_pt"][:, 0] > loose_pt) & (data["lep_pt"][:, 1] > loose_pt)]

        data["soft_pt"] = ak.min(data["lep_pt"], axis=1)

        four_momentum = vector.zip(
            {
                "pt": data["lep_pt"],
                "eta": data["lep_eta"],
                "phi": data["lep_phi"],
                "E": data["lep_e"],
            }
        )
        data["mass"] = (four_momentum[:, 0] + four_momentum[:, 1]).M
        data = data[(data["mass"] > 15) & (data["mass"] < 1000)]

        data["charge_product"] = data["lep_charge"][:, 0] * data["lep_charge"][:, 1]
        if want_os:
            data = data[data["charge_product"] < 0]
        elif want_ss:
            data = data[data["charge_product"] > 0]

        return data

    return preselection_cut


def write_preselection_parquet(
    analysis_parquet,
    read_variables=DEFAULT_READ_VARIABLES,
    string_codes=DEFAULT_STRING_CODES,
    base_output_dir="../../output-parquet/dilepton_ALLFLAVOUR_BASE_MediumID_v2",
):
    for sign in ("OS", "SS"):
        outdir = f"{base_output_dir}_{sign}"
        analysis_parquet(
            read_variables,
            string_codes,
            fraction=1,
            cut_function=make_preselection_cut(sign),
            write_parquet=True,
            output_directory=outdir,
            return_output=False,
        )
        print(f"[OK] Wrote {sign} parquet to: {outdir}")


def read_preselected(analysis_parquet, os_dir, ss_dir, scan_vars=SCAN_VARS):
    os_data = analysis_parquet(read_variables=scan_vars, read_directory=os_dir, return_output=True)
    ss_data = analysis_parquet(read_variables=scan_vars, read_directory=ss_dir, return_output=True)
    return os_data, ss_data


def channel_mask(arr, channel):
    lep0 = abs(arr["lep_type"][:, 0])
    lep1 = abs(arr["lep_type"][:, 1])

    if channel == "mumu":
        return (lep0 == 13) & (lep1 == 13) & arr["trigM"]
    if channel == "ee":
        return (lep0 == 11) & (lep1 == 11) & arr["trigE"]
    if channel in ("emu", "mue"):
        is_mixed = ((lep0 == 11) & (lep1 == 13)) | ((lep0 == 13) & (lep1 == 11))
        return is_mixed & (arr["trigE"] | arr["trigM"])

    raise ValueError("channel must be 'mumu', 'ee', or 'emu'")


def select_channel_dict(sample_dict, channel):
    return {key: arr[channel_mask(arr, channel)] for key, arr in sample_dict.items()}


def split_channels(all_os, all_ss):
    return {
        "mumu_os": select_channel_dict(all_os, "mumu"),
        "mumu_ss": select_channel_dict(all_ss, "mumu"),
        "ee_os": select_channel_dict(all_os, "ee"),
        "ee_ss": select_channel_dict(all_ss, "ee"),
        "emu_os": select_channel_dict(all_os, "emu"),
        "emu_ss": select_channel_dict(all_ss, "emu"),
    }


def pick_key(sample_dict, prefix):
    matches = [key for key in sample_dict if str(key).startswith(prefix)]
    if not matches:
        raise KeyError(f"No key starting with {prefix!r}. Available keys: {list(sample_dict)}")
    return matches[0]


def pick_key_optional(sample_dict, prefix):
    matches = [key for key in sample_dict if str(key).startswith(prefix)]
    return matches[0] if matches else None


def build_plot_dict(raw_dict, signal_prefix=None, signal_label=None, mc_prefixes=MC_PREFIXES):
    out = {"Data": raw_dict[pick_key(raw_dict, "2to4lep")]}

    if signal_prefix is not None:
        signal_key = pick_key_optional(raw_dict, signal_prefix)
        if signal_key is None:
            raise KeyError(f"Signal sample {signal_prefix!r} was not found.")
        out[signal_label or f"Signal {signal_prefix}"] = raw_dict[signal_key]

    for prefix in mc_prefixes:
        if prefix == signal_prefix:
            continue
        key = pick_key_optional(raw_dict, prefix)
        if key is not None:
            out[f"Background {prefix}"] = raw_dict[key]

    return out


def build_analysis_dicts(channels):
    return {
        "mumu_os": build_plot_dict(channels["mumu_os"], "Zmumu", "Signal Zmumu"),
        "mumu_ss": build_plot_dict(channels["mumu_ss"], "Zmumu", "Signal Zmumu"),
        "ee_os": build_plot_dict(channels["ee_os"], "Zee", "Signal Zee"),
        "ee_ss": build_plot_dict(channels["ee_ss"], "Zee", "Signal Zee"),
        "emu_os": build_plot_dict(channels["emu_os"]),
        "emu_ss": build_plot_dict(channels["emu_ss"]),
    }


def color_list_for(plot_dict):
    if len(plot_dict) > len(COLOR_CYCLE):
        raise ValueError(f"Need more colours for {len(plot_dict)} samples.")
    return COLOR_CYCLE[: len(plot_dict)]


def mass_mask(arr, mlow, mhigh):
    return (arr["mass"] > mlow) & (arr["mass"] < mhigh)


def iso_mask(arr, ptcut=None, etcut=None):
    mask = ak.ones_like(arr["mass"], dtype=bool)
    if ptcut is not None:
        mask = mask & (arr["lep_ptvarcone30"][:, 0] < ptcut) & (arr["lep_ptvarcone30"][:, 1] < ptcut)
    if etcut is not None:
        mask = mask & (arr["lep_topoetcone20"][:, 0] < etcut) & (arr["lep_topoetcone20"][:, 1] < etcut)
    return mask


def selected_mask(arr, ptcut=None, etcut=None, mlow=66, mhigh=116):
    return mass_mask(arr, mlow, mhigh) & iso_mask(arr, ptcut, etcut)


def apply_iso_cut(arr, ptcut=None, etcut=None):
    return arr[iso_mask(arr, ptcut, etcut)]


def apply_iso_to_plot_dict(plot_dict, ptcut=None, etcut=None):
    return {key: apply_iso_cut(arr, ptcut, etcut) for key, arr in plot_dict.items()}


def data_count_in_mass(arr, ptcut=None, etcut=None, mlow=66, mhigh=116):
    return int(ak.sum(selected_mask(arr, ptcut, etcut, mlow, mhigh)))


def weighted_yield_in_mass(arr, ptcut=None, etcut=None, mlow=66, mhigh=116, weight_var="totalWeight"):
    mask = selected_mask(arr, ptcut, etcut, mlow, mhigh)
    return float(ak.sum(arr[weight_var][mask]))


def background_keys(plot_dict):
    return [key for key in plot_dict if key.startswith("Background")]


def mc_keys(plot_dict):
    return [key for key in plot_dict if key != "Data"]


def yield_in_mass_for_keys(plot_dict, keys, ptcut=None, etcut=None, mlow=66, mhigh=116):
    total = 0.0
    components = {}
    for key in keys:
        if key not in plot_dict:
            continue
        value = weighted_yield_in_mass(plot_dict[key], ptcut, etcut, mlow, mhigh)
        components[key] = value
        total += value
    return total, components


def event_flow_counts(plot_dict, ptcut=None, etcut=None, mlow=66, mhigh=116):
    rows = []
    for sample, arr in plot_dict.items():
        mask_mass = mass_mask(arr, mlow, mhigh)
        mask_iso = iso_mask(arr, ptcut, etcut)
        mask_both = mask_mass & mask_iso
        row = {"sample": sample}
        if sample == "Data":
            row.update(
                {
                    "n_total": len(arr),
                    "n_after_mass": int(ak.sum(mask_mass)),
                    "n_after_iso": int(ak.sum(mask_iso)),
                    "n_after_mass_and_iso": int(ak.sum(mask_both)),
                }
            )
        else:
            row.update(
                {
                    "w_total": float(ak.sum(arr["totalWeight"])),
                    "w_after_mass": float(ak.sum(arr["totalWeight"][mask_mass])),
                    "w_after_iso": float(ak.sum(arr["totalWeight"][mask_iso])),
                    "w_after_mass_and_iso": float(ak.sum(arr["totalWeight"][mask_both])),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_df_delta(plot_dict_emu, ptcut=None, etcut=None, mlow=66, mhigh=116, scale=0.5):
    data_count = data_count_in_mass(plot_dict_emu["Data"], ptcut, etcut, mlow, mhigh)
    mc_yield, components = yield_in_mass_for_keys(
        plot_dict_emu,
        mc_keys(plot_dict_emu),
        ptcut,
        etcut,
        mlow,
        mhigh,
    )
    return {
        "emu_data": data_count,
        "emu_mc": mc_yield,
        "emu_mc_components": components,
        "df_scale": scale,
        "delta_df": scale * (data_count - mc_yield),
    }


def compute_ss_delta(plot_dict_os, plot_dict_ss, ptcut=None, etcut=None, mlow=66, mhigh=116):
    os_data = data_count_in_mass(plot_dict_os["Data"], ptcut, etcut, mlow, mhigh)
    ss_data = data_count_in_mass(plot_dict_ss["Data"], ptcut, etcut, mlow, mhigh)
    ss_mc, components = yield_in_mass_for_keys(
        plot_dict_ss,
        mc_keys(plot_dict_ss),
        ptcut,
        etcut,
        mlow,
        mhigh,
    )
    return {
        "os_data": os_data,
        "ss_data": ss_data,
        "ss_mc": ss_mc,
        "ss_mc_components": components,
        "delta_ss": ss_data - ss_mc,
        "corrected_os_data": os_data - (ss_data - ss_mc),
    }


def sigma_from_components(n_selected, n_background, eff, lumi_fb=36.6):
    if eff <= 0:
        return {"sigma_fb": np.nan, "unc_stat_fb": np.nan, "unc_lumi_fb": np.nan}
    sigma = (n_selected - n_background) / (eff * lumi_fb)
    unc_stat = np.sqrt(max(n_selected + n_background, 0.0)) / (eff * lumi_fb)
    return {
        "sigma_fb": sigma,
        "unc_stat_fb": unc_stat,
        "unc_lumi_fb": 0.017 * sigma,
    }


def run_zll_pipeline(config, ptcut=None, etcut=None, mlow=66, mhigh=116, apply_df=False, apply_ss=False, order=("iso", "df", "ss"), lumi_fb=36.6):
    os_dict = config.os_dict
    ss_dict = config.ss_dict
    emu_os = config.df_os_dict
    iso_applied = False
    df_info = None
    ss_info = None

    def current_cuts():
        return (None, None) if iso_applied else (ptcut, etcut)

    for step in order:
        if step == "iso":
            os_dict = apply_iso_to_plot_dict(os_dict, ptcut, etcut)
            ss_dict = apply_iso_to_plot_dict(ss_dict, ptcut, etcut)
            emu_os = apply_iso_to_plot_dict(emu_os, ptcut, etcut)
            iso_applied = True
        elif step == "df" and apply_df:
            cpt, cet = current_cuts()
            df_info = compute_df_delta(emu_os, cpt, cet, mlow, mhigh)
        elif step == "ss" and apply_ss:
            cpt, cet = current_cuts()
            ss_info = compute_ss_delta(os_dict, ss_dict, cpt, cet, mlow, mhigh)

    cpt, cet = current_cuts()
    n_selected = data_count_in_mass(os_dict["Data"], cpt, cet, mlow, mhigh)
    n_background_raw, background_components = yield_in_mass_for_keys(
        os_dict,
        background_keys(os_dict),
        cpt,
        cet,
        mlow,
        mhigh,
    )

    n_background = n_background_raw
    if apply_df and df_info is not None:
        n_background += df_info["delta_df"]
    if apply_ss and ss_info is not None:
        n_background += ss_info["delta_ss"]

    signal_yield = weighted_yield_in_mass(os_dict[config.signal_key], cpt, cet, mlow, mhigh)
    eff = signal_yield / config.produced_signal

    result = {
        "channel": config.channel,
        "ptcut": ptcut,
        "etcut": etcut,
        "mlow": mlow,
        "mhigh": mhigh,
        "order": " -> ".join(order),
        "apply_df": apply_df,
        "apply_ss": apply_ss,
        "N_selected": n_selected,
        "N_background_raw": n_background_raw,
        "N_background": n_background,
        "background_components": background_components,
        "signal_yield": signal_yield,
        "eff": eff,
        "df_info": df_info,
        "ss_info": ss_info,
        "delta_df": df_info["delta_df"] if df_info is not None else np.nan,
        "delta_ss": ss_info["delta_ss"] if ss_info is not None else np.nan,
        "df_data": df_info["emu_data"] if df_info is not None else np.nan,
        "df_mc": df_info["emu_mc"] if df_info is not None else np.nan,
        "ss_data": ss_info["ss_data"] if ss_info is not None else np.nan,
        "ss_mc": ss_info["ss_mc"] if ss_info is not None else np.nan,
    }
    result.update(sigma_from_components(n_selected, n_background, eff, lumi_fb))
    return result


def scan_zll_systematics(
    config,
    ptcuts,
    etcuts,
    mass_windows=((66, 116),),
    correction_modes=("none", "df", "ss", "df+ss"),
    orders=(("iso", "df", "ss"),),
    lumi_fb=36.6,
):
    rows = []
    total = len(ptcuts) * len(etcuts) * len(mass_windows) * len(correction_modes) * len(orders)
    with tqdm(total=total, desc=f"Scanning {config.channel}") as pbar:
        for mlow, mhigh in mass_windows:
            flow = None
            for ptcut in ptcuts:
                for etcut in etcuts:
                    flow = event_flow_counts(config.os_dict, ptcut, etcut, mlow, mhigh)
                    data_flow = flow[flow["sample"] == "Data"].iloc[0].to_dict()
                    for mode in correction_modes:
                        apply_df = mode in ("df", "df+ss")
                        apply_ss = mode in ("ss", "df+ss")
                        for order in orders:
                            result = run_zll_pipeline(
                                config,
                                ptcut=ptcut,
                                etcut=etcut,
                                mlow=mlow,
                                mhigh=mhigh,
                                apply_df=apply_df,
                                apply_ss=apply_ss,
                                order=order,
                                lumi_fb=lumi_fb,
                            )
                            rows.append(
                                {
                                    "channel": config.channel,
                                    "mlow": mlow,
                                    "mhigh": mhigh,
                                    "ptcut": ptcut,
                                    "etcut": etcut,
                                    "mode": mode,
                                    "order": result["order"],
                                    "sigma_fb": result["sigma_fb"],
                                    "unc_stat_fb": result["unc_stat_fb"],
                                    "unc_lumi_fb": result["unc_lumi_fb"],
                                    "N_selected": result["N_selected"],
                                    "N_background_raw": result["N_background_raw"],
                                    "N_background": result["N_background"],
                                    "signal_yield": result["signal_yield"],
                                    "eff": result["eff"],
                                    "delta_df": result["delta_df"],
                                    "delta_ss": result["delta_ss"],
                                    "df_data": result["df_data"],
                                    "df_mc": result["df_mc"],
                                    "ss_data": result["ss_data"],
                                    "ss_mc": result["ss_mc"],
                                    "data_n_total": data_flow["n_total"],
                                    "data_n_after_mass": data_flow["n_after_mass"],
                                    "data_n_after_iso": data_flow["n_after_iso"],
                                    "data_n_after_mass_and_iso": data_flow["n_after_mass_and_iso"],
                                }
                            )
                            pbar.update(1)
    return pd.DataFrame(rows)


def _max_two_to_numpy(arr, field):
    return np.maximum(
        ak.to_numpy(arr[field][:, 0]),
        ak.to_numpy(arr[field][:, 1]),
    )


def _array_to_numpy(arr, field):
    return ak.to_numpy(arr[field])


def _grid_yields(arr, ptcuts, etcuts, mass_windows, weighted):
    ptcuts = tuple(ptcuts)
    etcuts = tuple(etcuts)
    pt_bins = np.r_[-np.inf, np.asarray(ptcuts, dtype=float), np.inf]
    et_bins = np.r_[-np.inf, np.asarray(etcuts, dtype=float), np.inf]

    mass = _array_to_numpy(arr, "mass")
    pt_iso = _max_two_to_numpy(arr, "lep_ptvarcone30")
    et_iso = _max_two_to_numpy(arr, "lep_topoetcone20")
    weights = _array_to_numpy(arr, "totalWeight") if weighted else None

    grid = np.zeros((len(mass_windows), len(ptcuts), len(etcuts)), dtype=float)
    after_mass = np.zeros(len(mass_windows), dtype=float)

    all_hist, _, _ = np.histogram2d(pt_iso, et_iso, bins=(pt_bins, et_bins), weights=weights)
    after_iso = all_hist.cumsum(axis=0).cumsum(axis=1)[: len(ptcuts), : len(etcuts)]

    for iw, (mlow, mhigh) in enumerate(mass_windows):
        mass_sel = (mass > mlow) & (mass < mhigh)
        if weights is None:
            after_mass[iw] = float(np.sum(mass_sel))
            hist_weights = None
        else:
            after_mass[iw] = float(np.sum(weights[mass_sel]))
            hist_weights = weights[mass_sel]

        hist, _, _ = np.histogram2d(
            pt_iso[mass_sel],
            et_iso[mass_sel],
            bins=(pt_bins, et_bins),
            weights=hist_weights,
        )
        grid[iw] = hist.cumsum(axis=0).cumsum(axis=1)[: len(ptcuts), : len(etcuts)]

    return {
        "grid": grid,
        "after_mass": after_mass,
        "after_iso": after_iso,
        "total": float(len(arr) if weights is None else np.sum(weights)),
    }


def _sum_grid_summaries(plot_dict, keys, ptcuts, etcuts, mass_windows):
    total_grid = np.zeros((len(mass_windows), len(ptcuts), len(etcuts)), dtype=float)
    components = {}
    for key in keys:
        summary = _grid_yields(plot_dict[key], ptcuts, etcuts, mass_windows, weighted=True)
        components[key] = summary["grid"]
        total_grid += summary["grid"]
    return total_grid, components


def fast_scan_zll_systematics(
    config,
    ptcuts,
    etcuts,
    mass_windows=((66, 116),),
    correction_modes=("none", "df", "ss", "df+ss"),
    orders=(("iso", "df", "ss"),),
    lumi_fb=36.6,
):
    """Fast scan using cumulative pt/et isolation histograms.

    This is much faster than re-filtering the full Awkward Arrays for every
    ptcut/etcut point.  It assumes the scan order is the canonical final
    selection order; if multiple orders are passed, the same numerical result
    is reported for each because these operations commute for yield counting.
    """

    ptcuts = tuple(ptcuts)
    etcuts = tuple(etcuts)
    mass_windows = tuple(mass_windows)

    os_data = _grid_yields(config.os_dict["Data"], ptcuts, etcuts, mass_windows, weighted=False)
    os_signal = _grid_yields(config.os_dict[config.signal_key], ptcuts, etcuts, mass_windows, weighted=True)
    os_bkg_grid, _ = _sum_grid_summaries(config.os_dict, background_keys(config.os_dict), ptcuts, etcuts, mass_windows)

    ss_data = _grid_yields(config.ss_dict["Data"], ptcuts, etcuts, mass_windows, weighted=False)
    ss_mc_grid, _ = _sum_grid_summaries(config.ss_dict, mc_keys(config.ss_dict), ptcuts, etcuts, mass_windows)

    df_data = _grid_yields(config.df_os_dict["Data"], ptcuts, etcuts, mass_windows, weighted=False)
    df_mc_grid, _ = _sum_grid_summaries(config.df_os_dict, mc_keys(config.df_os_dict), ptcuts, etcuts, mass_windows)

    rows = []
    total = len(mass_windows) * len(ptcuts) * len(etcuts) * len(correction_modes) * len(orders)
    with tqdm(total=total, desc=f"Fast scanning {config.channel}") as pbar:
        for iw, (mlow, mhigh) in enumerate(mass_windows):
            for ip, ptcut in enumerate(ptcuts):
                for ie, etcut in enumerate(etcuts):
                    n_selected = os_data["grid"][iw, ip, ie]
                    n_background_raw = os_bkg_grid[iw, ip, ie]
                    signal_yield = os_signal["grid"][iw, ip, ie]
                    eff = signal_yield / config.produced_signal
                    delta_df = 0.5 * (df_data["grid"][iw, ip, ie] - df_mc_grid[iw, ip, ie])
                    delta_ss = ss_data["grid"][iw, ip, ie] - ss_mc_grid[iw, ip, ie]

                    for mode in correction_modes:
                        n_background = n_background_raw
                        if mode in ("df", "df+ss"):
                            n_background += delta_df
                        if mode in ("ss", "df+ss"):
                            n_background += delta_ss

                        sigma_info = sigma_from_components(n_selected, n_background, eff, lumi_fb)

                        for order in orders:
                            rows.append(
                                {
                                    "channel": config.channel,
                                    "mlow": mlow,
                                    "mhigh": mhigh,
                                    "ptcut": ptcut,
                                    "etcut": etcut,
                                    "mode": mode,
                                    "order": " -> ".join(order),
                                    "sigma_fb": sigma_info["sigma_fb"],
                                    "unc_stat_fb": sigma_info["unc_stat_fb"],
                                    "unc_lumi_fb": sigma_info["unc_lumi_fb"],
                                    "N_selected": n_selected,
                                    "N_background_raw": n_background_raw,
                                    "N_background": n_background,
                                    "signal_yield": signal_yield,
                                    "eff": eff,
                                    "delta_df": delta_df if mode in ("df", "df+ss") else np.nan,
                                    "delta_ss": delta_ss if mode in ("ss", "df+ss") else np.nan,
                                    "df_data": df_data["grid"][iw, ip, ie] if mode in ("df", "df+ss") else np.nan,
                                    "df_mc": df_mc_grid[iw, ip, ie] if mode in ("df", "df+ss") else np.nan,
                                    "ss_data": ss_data["grid"][iw, ip, ie] if mode in ("ss", "df+ss") else np.nan,
                                    "ss_mc": ss_mc_grid[iw, ip, ie] if mode in ("ss", "df+ss") else np.nan,
                                    "data_n_total": os_data["total"],
                                    "data_n_after_mass": os_data["after_mass"][iw],
                                    "data_n_after_iso": os_data["after_iso"][ip, ie],
                                    "data_n_after_mass_and_iso": n_selected,
                                }
                            )
                            pbar.update(1)

    return pd.DataFrame(rows)


def mode_table(config, ptcut=None, etcut=None, mlow=66, mhigh=116, order=("iso", "df", "ss"), lumi_fb=36.6):
    ptcut = config.nominal_ptcut if ptcut is None else ptcut
    etcut = config.nominal_etcut if etcut is None else etcut
    rows = []
    for mode in ("none", "df", "ss", "df+ss"):
        result = run_zll_pipeline(
            config,
            ptcut=ptcut,
            etcut=etcut,
            mlow=mlow,
            mhigh=mhigh,
            apply_df=mode in ("df", "df+ss"),
            apply_ss=mode in ("ss", "df+ss"),
            order=order,
            lumi_fb=lumi_fb,
        )
        rows.append({key: result[key] for key in [
            "channel",
            "sigma_fb",
            "N_selected",
            "N_background_raw",
            "N_background",
            "eff",
            "delta_df",
            "delta_ss",
            "df_data",
            "df_mc",
            "ss_data",
            "ss_mc",
        ]})
        rows[-1]["mode"] = mode
    columns = ["channel", "mode", "sigma_fb", "N_selected", "N_background_raw", "N_background", "eff", "delta_df", "delta_ss", "df_data", "df_mc", "ss_data", "ss_mc"]
    return pd.DataFrame(rows)[columns]


def systematic_from_variation(df, nominal_mask, value_col="sigma_fb"):
    sub = df.copy()
    nominal = sub[nominal_mask]
    if len(nominal) != 1:
        raise ValueError(f"nominal_mask must select exactly one row, selected {len(nominal)}.")
    sigma_nom = nominal.iloc[0][value_col]
    sub["delta_sigma"] = sub[value_col] - sigma_nom
    sub["abs_delta_sigma"] = np.abs(sub["delta_sigma"])
    sub["rel_delta_percent"] = 100.0 * sub["delta_sigma"] / sigma_nom
    sub["abs_rel_delta_percent"] = np.abs(sub["rel_delta_percent"])
    return sigma_nom, sub


def max_systematic(df_with_delta):
    return {
        "sys_abs_fb": float(df_with_delta["abs_delta_sigma"].max()),
        "sys_rel_percent": float(df_with_delta["abs_rel_delta_percent"].max()),
    }


def nominal_row(df, config):
    mlow, mhigh = config.nominal_mass_window
    return df[
        (df["channel"] == config.channel)
        & (df["mlow"] == mlow)
        & (df["mhigh"] == mhigh)
        & (df["ptcut"] == config.nominal_ptcut)
        & (df["etcut"] == config.nominal_etcut)
        & (df["mode"] == config.nominal_mode)
        & (df["order"] == config.nominal_order)
    ]


def compare_nominal_channels(scan_frames, configs):
    rows = []
    for df, config in zip(scan_frames, configs):
        selected = nominal_row(df, config)
        if len(selected) != 1:
            raise ValueError(f"Expected one nominal row for {config.channel}, found {len(selected)}")
        rows.append(selected.iloc[0])
    return pd.DataFrame(rows)


def print_sample_counts(label, plot_dict):
    print(f"\n{label}")
    for key, arr in plot_dict.items():
        print(f"{key:28s} {len(arr)}")


def plot_iso_contour(df_iso, nominal_ptcut, nominal_etcut, title, plt_module):
    pivot = df_iso.pivot_table(index="etcut", columns="ptcut", values="sigma_fb", aggfunc="mean")
    x = pivot.columns.values
    y = pivot.index.values
    x_grid, y_grid = np.meshgrid(x, y)
    z_grid = pivot.values

    plt_module.figure(figsize=(8, 6))
    contour = plt_module.contourf(x_grid, y_grid, z_grid, levels=20)
    plt_module.colorbar(contour, label="sigma_fb")
    lines = plt_module.contour(x_grid, y_grid, z_grid, levels=10)
    plt_module.clabel(lines, inline=True, fontsize=8)
    plt_module.scatter(nominal_ptcut, nominal_etcut, marker="x", s=100, label="Nominal cut")
    plt_module.xlabel("ptcut")
    plt_module.ylabel("etcut")
    plt_module.title(title)
    plt_module.legend()
    plt_module.tight_layout()
