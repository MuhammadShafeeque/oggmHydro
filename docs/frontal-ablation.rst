Frontal ablation
================

OGGM has a simple parameterization for frontal ablation that can be used
during the ice volume estimation and for dynamical runs. These features
are still in the testing phase and are switched off per default.

Visit our `tutorials <https://tutorials.oggm.org/stable/notebooks/tutorials/kcalving_parameterization.html>`_
if you are interested and would like to give them a go.

.. figure:: https://oggm.org/img/blog/calving_param/calving_ex.png
    :width: 100%
    :align: left

    Illustration of the water-depth – calving-rate feedbacks. See the tutorials
    for more details.

Ocean-forced frontal ablation
-----------------------------

The calving constant of this parameterization can be made to depend on the
ocean. :py:func:`tasks.process_ocean_data` writes an ``ocean_data`` file to
the glacier directory (the thermal forcing in one or more depth bands, the
sea-ice concentration and, optionally, the subglacial discharge), and
:py:func:`tasks.run_with_ocean_forcing` runs the flowline model with one of
four calving laws, chosen with ``PARAMS['calving_law']``:

- ``constant``: the default law, :math:`q = k \, d \, h \, w`
- ``tf_power``: :math:`k(t) = k_0 \, (TF(t) / TF_{ref})^{\gamma}`, with
  :math:`TF` the thermal forcing
- ``melt_calving``: a submarine melt rate after Rignot et al. (2016) added to
  the calving term, :math:`q = w \, d \, (k_c \, h + \lambda \, \dot{m})`
- ``sea_ice``: ``melt_calving`` with the melt term multiplied by
  :math:`f_{ow}^{\delta}`, :math:`f_{ow}` being the open-water fraction

The laws are nested: each reduces to the one before it for a value of its
parameters (:math:`TF = TF_{ref}` or :math:`\gamma = 0`, :math:`\lambda = 0`,
:math:`\delta = 0`). A calving constant calibrated against an observed frontal
ablation already contains the melt-driven part, so the melt term of the last
two laws adds to it unless the law is built with ``partition=True``
(:py:func:`core.ocean_calving.ocean_calving_law`).

For the inversion, the water depth at the calving front can be taken from a
measured bed instead of being solved for: add the BedMachine bed to the
glacier directory (:py:func:`shop.bedmachine_bed.bedmachine_bed_to_gdir`), run
:py:func:`tasks.terminus_water_depth_from_bed`, and set
``PARAMS['inversion_calving_from_bathymetry']``. The calving constant can then
be fitted to observed frontal ablation before the mass balance is calibrated
(:py:func:`core.ocean_inversion.fit_calving_k`), and
:py:func:`tasks.mb_calibration_from_geodetic_mb` calibrates the surface mass
balance against the geodetic mass change plus that frontal ablation when
``gdir.inversion_calving_rate`` is set. After ``init_present_time_glacier``,
:py:func:`tasks.bedmachine_terminus_bed` puts the same water depth on the
flowline of the run.

Like the calving parameterization itself, these features are experimental.
