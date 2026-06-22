from __future__ import annotations

from pathlib import Path


def make_requested_plots(run_config, observables: list[dict], output_dir: str | Path) -> None:
    if not run_config.outputs.plots.enabled or not observables:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError('Plot output requires the optional matplotlib dependency. Install plasma-global-model[plot].') from exc

    output_dir = Path(output_dir)
    items = set(run_config.outputs.plots.items or [])
    formats = list(run_config.outputs.plots.format or ['png'])
    dpi = int(run_config.outputs.plots.dpi)

    t = [row['time_s'] for row in observables]
    mapping = {
        'electron_density_vs_time': ('electron_density_m3', 'Electron density [m^-3]'),
        'mean_energy_vs_time': ('mean_electron_energy_eV', 'Mean electron energy [eV]'),
        'self_bias_vs_time': ('self_bias_V', 'Self-bias [V]'),
    }
    for item, (key, ylabel) in mapping.items():
        if item not in items or key not in observables[0]:
            continue
        y = [row[key] for row in observables]
        plt.figure()
        plt.plot(t, y)
        plt.xlabel('time [s]')
        plt.ylabel(ylabel)
        plt.tight_layout()
        for ext in formats:
            plt.savefig(output_dir / f'{item}.{ext}', dpi=dpi)
        plt.close()
