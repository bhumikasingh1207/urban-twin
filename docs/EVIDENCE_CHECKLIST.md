# Evidence checklist (tick only what you have actually produced)

For each item record: date, command used, and file/screenshot name. Do not include synthetic-data outputs.

- [ ] Repository link (public or accessible to evaluators): ______
- [ ] `pytest -q` output screenshot (shows tests pass on your machine)
- [ ] Study area approved by guide (date): ______  `config.yaml` screenshot
- [ ] Data acquisition log (scene dates, cloud %, tile IDs) from `acquire`
- [ ] `outputs/real/preprocessing_report.json` (missing-value report)
- [ ] True-colour composite screenshot (dashboard map, real data)
- [ ] Land-use classification map screenshot, with legend
- [ ] `outputs/real/metrics.json` + confusion matrix PNGs
- [ ] Number of independent validation points per class: ______ (source/method described)
- [ ] Area statistics table (`area_stats.csv`) and chart
- [ ] Change map + transition matrix (`transition_matrix_km2.csv`)
- [ ] Growth model metrics (`growth_metrics.json`) and scenario screenshot (parameters visible)
- [ ] NDVI / LST page (only if the LST step actually succeeded)
- [ ] Dashboard walkthrough screenshots (all pages)
- [ ] Demonstration video link (accessible): ______
- [ ] Error log: problems met and how each was fixed

Keep apart in the report: literature findings · implemented features · partial features · future work ·
your own experimental results.
