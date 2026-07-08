from __future__ import annotations

from typing import Any

from plasma_global.physics.wall_loss import (
    IonLossConfigError,
    zone_effective_frequency_surface_counts,
    zone_ion_loss_properties,
)
from plasma_global.validation import ValidationIssue


def validate_surface_ion_loss_config(chamber: Any) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    try:
        zone_ion_loss_properties(chamber)
        frequency_counts = zone_effective_frequency_surface_counts(chamber)
    except IonLossConfigError as exc:
        issues.append(ValidationIssue('ERROR', exc.code, str(exc), exc.entity_id))
        return issues
    except Exception as exc:
        issues.append(ValidationIssue('ERROR', 'SURFACE_ION_LOSS_CONFIG_INVALID', str(exc)))
        return issues

    for zone_id, count in frequency_counts.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    'WARNING',
                    'ION_LOSS_FREQUENCY_MULTIPLE_SURFACES',
                    f'Zone {zone_id} has multiple effective-frequency ion wall-loss surfaces; the solver uses an area-weighted rate.',
                    zone_id,
                )
            )
    return issues
