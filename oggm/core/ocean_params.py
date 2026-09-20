"""Parameter defaults for the ocean-forced frontal ablation extension.

Kept in their own module because both the data side (``oggm.shop.ocean``) and the
model side (``oggm.core.ocean_calving``, ``oggm.core.ocean_inversion``) read them,
and because ``cfg.initialize()`` rewrites ``cfg.PARAMS`` wholesale: these defaults
are therefore *not* written into ``oggm/params.cfg``, which is upstream's file.

Call :func:`init_ocean_params` after every ``cfg.initialize()``, or read through
:func:`ocean_param`, which falls back to the defaults here when the key is absent.
"""
import logging

from oggm import cfg
from oggm.exceptions import InvalidParamsError

log = logging.getLogger(__name__)

# The three candidate depth bands. 'terminus' is a placeholder bottom: it is replaced
# per glacier by the measured or inverted water depth, which is the whole point of
# carrying a shallow band at a site where the terminus sits in ~60 m of water.
# 'ismip6' is Slater et al. (2020); 'moller' is Möller et al. (2024).
DEFAULTS = {
    'ocean_depth_bands': [('terminus', 0., 100.),
                          ('ismip6', 200., 500.),
                          ('moller', 0., 700.)],
    'ocean_band_weighting': {'terminus': 'uniform',
                             'ismip6': 'uniform',
                             'moller': 'depth_weighted'},
    'ocean_tf_band': 'terminus',
    'ocean_tf_ref': None,
    'ocean_tf_ref_period': ('1995', '2014'),
    'ocean_use_teos10': False,
    'ocean_n_cells': 10,
    'ocean_search_radius_km': 25.,
    'ocean_open_water_threshold': 0.15,
    'ocean_bias_correct': False,
    'ocean_water_density': 1028.,

    'calving_law': 'constant',
    'calving_tf_exponent': 1.18,
    'calving_melt_A': 3e-4,
    'calving_melt_B': 0.15,
    'calving_melt_alpha': 0.39,
    'calving_melt_beta': 1.18,
    'calving_undercut_efficiency': 1.0,
    'calving_openwater_exponent': 1.0,
    'calving_ocean_water_depth': 'inverted',

    # --- the bed under the calving front (oggm.shop.bedmachine_bed,
    #     oggm.core.bedmachine_flowline) ---
    'bedmachine_version': None,
    'bedmachine_file': None,
    'bed_extension_width_method': 'inversion',
    'bed_extension_match_terminus': False,
    'calving_front_width_rtol': 0.05,
}


def init_ocean_params(reset=False):
    """Add the ocean parameters to ``cfg.PARAMS``.

    Existing values are kept unless ``reset`` is True, so a user override set before
    this call survives it.
    """
    # These are extension defaults, not user parameter changes, and PARAMS logs a
    # warning for every key it does not already know.
    do_log, cfg.PARAMS.do_log = cfg.PARAMS.do_log, False
    try:
        for k, v in DEFAULTS.items():
            if reset or k not in cfg.PARAMS:
                cfg.PARAMS[k] = v
    finally:
        cfg.PARAMS.do_log = do_log


def ocean_param(key):
    """One ocean parameter, from ``cfg.PARAMS`` if set and from DEFAULTS otherwise."""
    if key not in DEFAULTS:
        raise InvalidParamsError(f'{key!r} is not an ocean parameter')
    return cfg.PARAMS.get(key, DEFAULTS[key])
