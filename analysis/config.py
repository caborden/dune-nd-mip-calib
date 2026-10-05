"""Common settings for standalone and orchestrated analyses."""
import numpy as np

def make_settings(segment_length_cm=3., step_cm=3., grouping='io-group', geometry=None,
                  require_through_going=False, face_cuts='auto', face_margin_cm=1.,
                  bins=100, hist_range=(0.,90.), fit_range=(20.,50.), min_fit_entries=100):
    if grouping not in {'io-group', 'legacy-tpc'}:
        raise ValueError('grouping must be io-group or legacy-tpc')
    if face_cuts not in {'auto','none'}:
        raise ValueError('face_cuts must be auto or none')
    if not (np.isfinite([segment_length_cm,step_cm,face_margin_cm,*hist_range,*fit_range]).all()
            and segment_length_cm > 0 and step_cm > 0 and face_margin_cm >= 0
            and bins > 4 and min_fit_entries > 0
            and hist_range[0] < fit_range[0] < fit_range[1] < hist_range[1]):
        raise ValueError('Invalid segment scales, histogram bins, or fit/histogram ranges')
    if require_through_going and not geometry:
        raise ValueError('--require-through-going requires --geometry-json')
    config = dict(segment_length_cm=segment_length_cm,step_cm=step_cm,grouping=grouping,
                  geometry=geometry or {},require_through_going=require_through_going,
                  face_margin_cm=face_margin_cm,face_cuts=face_cuts,bins=bins,hist_range=list(hist_range),
                  fit_range=list(fit_range),min_fit_entries=min_fit_entries,
                  fit_model='upstream Moyal-like Landau approximation convolved with Gaussian',
                  charge_field='Q',charge_units_assumed='ke−',hit_view='selected only',
                  boundary_policy='upstream inclusive endpoints; exact boundary hits can appear in adjacent windows')
    return config

SEGMENT_KEYS = {"segment_length_cm", "step_cm", "grouping", "geometry",
                "require_through_going", "face_cuts", "face_margin_cm"}
