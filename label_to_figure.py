# !/usr/bin/python3
""" This module defines a dictionnary that link the id of LaTeX figures to their labels. It can be automatically
generated by LaTeX.
You can then look for this function name in the file figures.py """
FIGURE = {}
FIGURE["4.6"] = "fig_alpha_sl"
FIGURE["4.7"] = "fig_neutral_comparisonPlot"
FIGURE["4.9"] = "fig_mixing_lengths"
FIGURE["4.10"] = "fig_sensitivity_delta_sl"
FIGURE["4.12"] = "fig_Stratified"
FIGURE["4.13"] = "fig_consistency_comparisonStratified"
FIGURE["4.14"] = "fig_consistency_comparisonUnstable"

FIGURE["AMAC_pres1"] = "fig_compareASLsize_ustar"
FIGURE["AMAC_pres2"] = "fig_compareASLsize"

FIGURE["10mai_1"] = "fig_referencefrictionScales"
FIGURE["10mai_2"] = "fig_referenceCoupling"
ALL_LABELS = FIGURE
