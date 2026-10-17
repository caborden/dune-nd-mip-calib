"""Collective plots, numerical CSVs, and a portable HTML/PDF report."""
import csv
import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import numpy as np

from .flow import ADDRESS, json_value

LABELS = {
    'charge': 'Calibrated hit Q', 'adc': 'Raw data-packet ADC [dataword]',
    'drift_us': 'Hit drift time [us]', 'event_hits': 'Unique linked hit rows / event',
    'event_channels': 'Unique electronics channels / event',
    'event_charge': 'Sum of linked hit Q / event',
    'event_duration_us': 'Linked-hit time span within event [us]',
    'multiplicity': 'Hits / channel / event',
    'delta_us': 'Successive same-channel hit separation within event [us]',
}


def write_json(path, value):
    path.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + '\n')


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ratio(numerator, denominator):
    result = np.full(np.shape(numerator), np.nan, dtype=float)
    np.divide(numerator, denominator, out=result, where=np.asarray(denominator) > 0)
    return result


def draw_ratio(ax, centers, values, reference_counts, settings, color):
    """Bound only the display. Preserve extremes with boundary triangles.

    Hollow symbols mark sparse reference bins, not a significance test or a
    rejection. Numerator zeros are retained; zero denominators remain undefined.
    """
    low, high = settings['ratio_ylim']
    finite = np.isfinite(values)
    sparse = reference_counts < settings['ratio_min_reference_count']
    below, above = finite & (values < low), finite & (values > high)
    within = finite & ~below & ~above
    for selected, marker in ((within, 'o'), (below, 'v'), (above, '^')):
        for hollow in (False, True):
            mask = selected & (sparse == hollow)
            ax.scatter(centers[mask], np.clip(values[mask], low, high),
                       s=12 if marker == 'o' else 25, marker=marker,
                       facecolors='none' if hollow else color, edgecolors=color,
                       linewidths=.7, clip_on=False, zorder=3)
    ax.set_ylim(low, high)
    return dict(undefined=int((~finite).sum()), low_reference_count=int((finite & sparse).sum()),
                below_display=int(below.sum()), above_display=int(above.sum()))


def sample_summary(sample):
    warnings = list(sample.warnings)
    for name, hist in sample.histograms.items():
        if len(sample.coverage[name]) != len(sample.files):
            warnings.append(f'{name}: fields available in {len(sample.coverage[name])}/{len(sample.files)} files.')
        if hist.total and (hist.underflow+hist.overflow)/hist.total > .01:
            warnings.append(f'{name}: more than 1% of entries lie outside the plotted range; consider wider bins.')
    for key, value in sample.audit.items():
        if value and any(term in key for term in ('invalid', 'nonfinite', 'over_link_limit', 'duplicate_event')):
            warnings.append(f'{key}: {value}')
    bad_positions = sum(not r['position_stable'] for r in sample.channel_rows())
    if bad_positions:
        warnings.append(f'{bad_positions} observed channels lack stable finite y/z positions; omitted from channel maps.')
    envelopes = [(f['path'], f.get('event_unix_envelope')) for f in sample.files]
    for i, (path, span) in enumerate(envelopes):
        if span is None:
            continue
        for other, other_span in envelopes[:i]:
            if other_span and max(span[0], other_span[0]) <= min(span[1], other_span[1]):
                warnings.append(f'Event timestamp envelopes overlap: {path} and {other}. '
                                'This is not proof of duplicated data; check acquisition provenance.')
    fraction_pairs = {
        'nonfinite_charge_hit_fraction': ('nonfinite_charge_hits', 'hit_rows'),
        'nonpositive_charge_hit_fraction': ('nonpositive_charge_hits', 'hit_rows'),
        'disabled_fraction_where_flag_available': ('disabled_flagged_hits', 'hits_with_disabled_flag_available'),
        'adc_endpoint_fraction': ('adc_at_configured_max', 'data_packet_rows'),
        'bad_parity_fraction_where_available': ('data_packets_invalid_parity', 'data_packets_with_parity_available'),
        'analyzed_event_fraction': ('events_analyzed', 'event_rows'),
        'empty_fraction_of_analyzed_events': ('empty_events', 'events_analyzed'),
    }
    fractions = {name: sample.audit[numerator]/sample.audit[denominator]
                 if sample.audit[denominator] else None
                 for name, (numerator, denominator) in fraction_pairs.items()}
    return dict(name=sample.name, label=sample.label, audit=dict(sample.audit), fractions=fractions,
                observed_channels=len(sample.channels), histograms={
                    k: dict(v.summary(), files_with_required_fields=len(sample.coverage[k]))
                    for k, v in sample.histograms.items()},
                projections={k: {n: v for n, v in p.items() if n != 'counts'}
                             for k, p in sample.projections.items()},
                files=sample.files, warnings=warnings)


def render(samples, axes, output, manifest):
    plots, tables = output/'plots', output/'tables'
    plots.mkdir()
    tables.mkdir()
    figures = []
    summaries = [sample_summary(s) for s in samples]
    manifest['samples'] = summaries
    reference = samples[0]
    with PdfPages(output/'report.pdf') as pdf:
        def save(fig, name, caption):
            fig.savefig(plots/(name+'.png'), dpi=140, bbox_inches='tight')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            figures.append((name, caption))

        for key in LABELS:
            label = LABELS[key]
            if key in ('charge', 'event_charge'):
                label += ' [' + samples[0].settings['charge_unit'] + ']'
            fig, (ax, lower) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                           gridspec_kw={'height_ratios': [3, 1]}, layout='constrained')
            ref = reference.histograms[key]
            ref_fraction = ref.counts / max(ref.total, 1)
            display_audit = {}
            for sample_index, sample in enumerate(samples):
                hist = sample.histograms[key]
                fraction = hist.counts / max(hist.total, 1)
                coverage = f'fields: {len(sample.coverage[key])}/{len(sample.files)} inputs'
                color = f'C{sample_index % 10}'
                ax.stairs(fraction, hist.edges,
                          label=f'{sample.label}: N={hist.total:,}; {coverage}', linewidth=1.5, color=color)
                centers = (hist.edges[1:]+hist.edges[:-1])/2
                values = ratio(fraction, ref_fraction)
                if sample is not reference and hist.total:
                    display_audit[sample.name] = draw_ratio(lower, centers, values, ref.counts,
                                                          sample.settings, color)
                write_csv(tables/f'{sample.name}_{key}.csv', [
                    dict(low=lo, high=hi, count=int(count), fraction=float(frac),
                         ratio_to_reference=float(rat), reference_count=int(ref_count),
                         ratio_low_reference_count=bool(ref_count < sample.settings['ratio_min_reference_count']),
                         ratio_outside_display=bool(np.isfinite(rat) and not
                                                    sample.settings['ratio_ylim'][0] <= rat <= sample.settings['ratio_ylim'][1]))
                    for lo, hi, count, frac, rat, ref_count in zip(hist.edges[:-1], hist.edges[1:],
                                                                  hist.counts, fraction, values, ref.counts)])
            manifest.setdefault('ratio_display', {})[key] = display_audit
            if any(np.any(s.histograms[key].counts) for s in samples):
                ax.set_yscale('log')
            else:
                ax.text(.5, .5, 'No available in-range entries', ha='center', transform=ax.transAxes)
            ax.set(ylabel='Fraction of entries / bin', title=label)
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
            lower.axhline(1, color='gray', lw=1)
            lower.set_ylim(*reference.settings['ratio_ylim'])
            lower.set(xlabel=label, ylabel='Ratio to\n'+reference.name)
            lower.grid(alpha=.2)
            if display_audit:
                clipped = sum(v['above_display'] + v['below_display'] for v in display_audit.values())
                lower.text(.99, .96, f'Triangles: {clipped} off-scale bins; open: reference < '
                           f'{reference.settings["ratio_min_reference_count"]}',
                           transform=lower.transAxes, ha='right', va='top', fontsize=7,
                           bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
            else:
                lower.text(.5, .5, 'No comparison entries', ha='center', transform=lower.transAxes)
            caption = ('Counts pooled across files before normalization. N includes nonfinite and out-of-range '
                       'entries supplied to this histogram; displayed fractions may sum to less than one. '
                       'Undefined ratios are omitted. Ratio limits affect display only: triangles mark off-scale '
                       'values and open symbols mark sparse reference bins. All finite ratios remain in the CSV. '
                       'Ratios are descriptive, with no independence assumption.')
            if key == 'adc':
                caption += ' ADC uses all packet_type=0 rows, not only packets associated to calibrated hits.'
            if key == 'event_duration_us':
                caption += ' This is the observed hit span, not the configured event-builder window.'
            save(fig, key, caption)

        # Common spatial ranges AND common color normalization for every sample.
        for pair in ('xy', 'xz', 'yz'):
            fractions = [s.projections[pair]['counts']/max(s.projections[pair]['total'], 1) for s in samples]
            positive = np.concatenate([a[a > 0] for a in fractions])
            norm = LogNorm(vmin=float(positive.min()), vmax=max(float(positive.max()), float(positive.min())*1.01)) if len(positive) else None
            fig, panels = plt.subplots(1, len(samples), figsize=(6*len(samples), 5), squeeze=False, layout='constrained')
            for sample, values, ax in zip(samples, fractions, panels[0]):
                item = sample.projections[pair]
                mesh = ax.pcolormesh(axes[pair[0]], axes[pair[1]], np.ma.masked_less_equal(values.T, 0), norm=norm, cmap='viridis')
                ax.set(xlabel=f'{pair[0]} [cm]', ylabel=f'{pair[1]} [cm]',
                       title=f'{sample.label}\nCoordinates: {item["files"]}/{len(sample.files)} inputs')
                ax.set_aspect('equal')
                fig.colorbar(mesh, ax=ax, label='Fraction of hit rows / bin')
                np.savez_compressed(tables/f'{sample.name}_{pair}.npz', counts=item['counts'],
                                    first_edges=axes[pair[0]], second_edges=axes[pair[1]],
                                    normalization_hits=item['total'])
            save(fig, 'projection_'+pair, 'Inclusive calibrated hits; common axes/color scale. '
                 'Ranges use finite data extents unless spatial_ranges is configured. '
                 'Nonfinite/out-of-range coordinates are counted in summary.json.')

        channel_tables = [dict((tuple(row[n] for n in ADDRESS), row) for row in s.channel_rows()) for s in samples]
        for sample, rows in zip(samples, channel_tables):
            write_csv(tables/f'{sample.name}_channels.csv', rows.values(), fields=list(ADDRESS)+[
                'count', 'hit_fraction', 'y', 'z', 'position_stable', 'median_charge', 'median_status',
                'finite_charge', 'charge_underflow', 'charge_overflow'])
        groups = sorted({key[0] for rows in channel_tables for key in rows})
        for group in groups:
            group_rows = [[r for key, r in table.items() if key[0] == group and r['position_stable']]
                          for table in channel_tables]
            for metric in ('hit_fraction', 'median_charge'):
                finite = [r[metric] for rows in group_rows for r in rows if np.isfinite(r[metric])]
                low, high = (min(finite), max(finite)) if finite else (0, 1)
                if low == high:
                    high = low+max(abs(low)*.01, 1e-6)
                norm = LogNorm(low, high) if metric == 'hit_fraction' and low > 0 else None
                fig, panels = plt.subplots(1, len(samples), figsize=(6*len(samples), 5), squeeze=False, layout='constrained')
                positions = [r for rows in group_rows for r in rows]
                for sample, rows, ax in zip(samples, group_rows, panels[0]):
                    valid = [r for r in rows if np.isfinite(r[metric])]
                    mesh = ax.scatter([r['z'] for r in valid], [r['y'] for r in valid],
                                      c=[r[metric] for r in valid], s=15, marker='s', cmap='viridis',
                                      **({'norm': norm} if norm else dict(vmin=low, vmax=high)))
                    ax.set(xlabel='z [cm]', ylabel='y [cm]', title=f'{sample.label}\nIO group {group}')
                    ax.set_aspect('equal')
                    set_map_limits(ax, positions)
                    fig.colorbar(mesh, ax=ax, label='Fraction of all sample hits' if metric == 'hit_fraction'
                                 else f'Binned median Q [{samples[0].settings["charge_unit"]}]')
                save(fig, f'channels_iog{group}_{metric}',
                     'Only observed channels with stable finite readout-plane positions are drawn. '
                     'Blank regions do not distinguish disabled, unmapped, or enabled-but-unobserved channels. '
                     'Charge medians pool fine charge histograms; medians outside that range are omitted.')

        for sample, table in zip(samples[1:], channel_tables[1:]):
            rows = channel_ratios(channel_tables[0], table)
            write_csv(tables/f'{sample.name}_over_{reference.name}_channels.csv', rows)
            for group in groups:
                selected = [r for r in rows if r['io_group'] == group and r['position_compatible']]
                valid = [r for r in selected if np.isfinite(r['ratio'])]
                absent = [r for r in selected if not np.isfinite(r['ratio'])]
                fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')
                mesh = ax.scatter([r['z'] for r in valid], [r['y'] for r in valid],
                                  c=[r['ratio'] for r in valid], s=18, marker='s',
                                  cmap='coolwarm', vmin=0, vmax=2)
                if absent:
                    ax.scatter([r['z'] for r in absent], [r['y'] for r in absent], marker='x',
                               color='gray', s=15, label='No reference hits')
                    ax.legend()
                ax.set(xlabel='z [cm]', ylabel='y [cm]',
                       title=f'{sample.label} / {reference.label}\nOccupancy fraction ratio; IO group {group}')
                ax.set_aspect('equal')
                set_map_limits(ax, selected)
                fig.colorbar(mesh, ax=ax, label='Ratio of sample-wide hit fractions', extend='max')
                save(fig, f'{sample.name}_over_{reference.name}_iog{group}',
                     'Channel addresses matched across samples. Zero means no numerator hits; gray crosses mean '
                     'no reference hits. Values above 2 saturate the color scale; CSV retains full values. '
                     'Channels whose positions disagree are omitted. No channel-mask equivalence is assumed.')

    write_json(output/'summary.json', manifest)
    warning_blocks = ''.join('<h3>'+html.escape(s['label'])+'</h3><ul>'+''.join(
        '<li>'+html.escape(w)+'</li>' for w in s['warnings'])+'</ul>' for s in summaries)
    counts = ''.join('<tr><td>'+html.escape(s['label'])+'</td>'+''.join(
        f'<td>{v:,}</td>' for v in (len(s['files']), s['audit'].get('hit_rows', 0),
                                   s['audit'].get('event_rows', 0), s['audit'].get('events_analyzed', 0),
                                   s['observed_channels']))+'</tr>' for s in summaries)
    audit_names = sorted({name for s in summaries for name in s['audit']})
    audit_table = '<table><tr><th>Counter</th>'+''.join('<th>'+html.escape(s['label'])+'</th>'
                  for s in summaries)+'</tr>'+''.join(
        '<tr><td>'+html.escape(name)+'</td>'+''.join('<td>'+str(s['audit'].get(name, '—'))+'</td>'
                                                  for s in summaries)+'</tr>' for name in audit_names)+'</table>'
    histogram_table = ('<table><tr><th>Sample / metric</th><th>N</th><th>Underflow</th>'
                       '<th>Overflow</th><th>Nonfinite</th><th>Files with fields</th></tr>')
    for s in summaries:
        for name, item in s['histograms'].items():
            histogram_table += '<tr><td>'+html.escape(s['name']+' / '+name)+'</td>'+''.join(
                '<td>'+str(item[k])+'</td>' for k in ('total', 'underflow', 'overflow', 'nonfinite',
                                                     'files_with_required_fields'))+'</tr>'
    histogram_table += '</table>'
    cards = ''.join(f'<section><h2>{html.escape(name)}</h2><img src="plots/{name}.png" alt="{name}">'
                    f'<p>{html.escape(caption)}</p></section>' for name, caption in figures)
    (output/'report.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Inclusive FLOW data quality</title><style>
body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:0 20px;color:#20252b}
img{max-width:100%}section{border-top:1px solid #ddd;margin-top:35px}td,th{padding:8px;text-align:left}
table{border-collapse:collapse}tr{border-bottom:1px solid #ddd}li{overflow-wrap:anywhere}</style>
<h1>Inclusive FLOW sample comparison</h1><p>No track selection. Inputs pooled within each sample.
Stored Q values are interpreted as '''+html.escape(samples[0].settings['charge_unit'])+'''
without numerical rescaling. No live-time or lifetime correction is applied.</p>
<p><a href="report.pdf">Plot PDF</a> · <a href="summary.json">Metadata and integrity audit</a> ·
<a href="tables/">Numerical tables</a></p>
<table><tr><th>Sample</th><th>Files</th><th>Hit rows</th><th>Event rows</th>
<th>Events analyzed</th><th>Observed channels</th></tr>'''+counts+
        '</table><h2>Coverage and diagnostics</h2>'+warning_blocks+
        '<details><summary>Integrity counters (— means unavailable or not encountered)</summary>'+audit_table+
        '</details><details><summary>Histogram denominators and overflow</summary>'+histogram_table+
        '</details>'+cards+'</html>')


def set_map_limits(ax, rows):
    if rows:
        z, y = [r['z'] for r in rows], [r['y'] for r in rows]
        ax.set_xlim(min(z)-.5, max(z)+.5)
        ax.set_ylim(min(y)-.5, max(y)+.5)


def channel_ratios(reference, current):
    result = []
    for key in sorted(reference.keys() | current.keys()):
        a, b = reference.get(key), current.get(key)
        position = a if a and a['position_stable'] else b
        compatible = bool(position and position['position_stable'])
        if a and b:
            compatible = bool(a['position_stable'] and b['position_stable'] and
                              np.allclose([a['y'], a['z']], [b['y'], b['z']], atol=1e-3, rtol=0))
        denominator, numerator = a['hit_fraction'] if a else 0, b['hit_fraction'] if b else 0
        result.append(dict(zip(ADDRESS, key), reference_count=a['count'] if a else 0,
                           numerator_count=b['count'] if b else 0, reference_fraction=denominator,
                           numerator_fraction=numerator, ratio=numerator/denominator if denominator else np.nan,
                           y=position['y'] if position else np.nan, z=position['z'] if position else np.nan,
                           position_compatible=compatible))
    return result
