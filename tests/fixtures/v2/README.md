# Schema-v2 migration fixtures

These inputs are retained only to test the one-way `migrate-v2` boundary. They
are not runtime examples and schema v2 is never accepted by `load_case`.

The five cases exercise the former smoke, Argon LXCat, CRANE, RF-envelope, and
ZDPlaskin configurations. Argon LXCat conversion explicitly uses
`tools/importers/argon_lxcat_cross_section_segments.yaml`; ZDPlaskin wall
neutralization explicitly uses
`tools/importers/zdplaskin_boundary_products.yaml`. The migrator must fail
rather than guess when equivalent mappings are absent for ambiguous source
data.
