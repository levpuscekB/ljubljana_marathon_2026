"""Historical Ljubljana Marathon archive (1996-2026, excluding the virtual 2020 event)."""
from urllib.parse import urlencode

EVENT_YEARS = tuple(list(range(1996, 2020)) + list(range(2021, 2027)))
# Numbered editions skipped 2020; virtual 2020 results have no chip times.
LM_EDITION = {y: (y-1995 if y <= 2019 else y-1996) for y in EVENT_YEARS}
ALL_SPLITS = [5, 10, 15, 20, 30]
RACES = {
    '10': {'label': '10 km · Garmin', 'long_name': 'Garmin 10 km', 'km': 10.0,
           'disc': {'M': 'RM', 'F': 'RZ'}, 'checkpoints': [5], 'age_disc': {}},
    '21': {'label': '21.1 km · Half marathon', 'long_name': 'Half marathon', 'km': 21.0975,
           'disc': {'M': '21M', 'F': '21Z'}, 'checkpoints': [5, 10, 15, 20], 'age_disc': {}},
    '42': {'label': '42.2 km · Marathon', 'long_name': 'Marathon', 'km': 42.195,
           'disc': {'M': '42M', 'F': '42Z'}, 'checkpoints': [10, 20, 30], 'age_disc': {}},
}
RACE_DATES = {2025: {'10': '2025-10-18', '21': '2025-10-19', '42': '2025-10-19'},
              2026: {'10': '2026-10-17', '21': '2026-10-18', '42': '2026-10-18'}}


def race_date(year, race):
    return RACE_DATES.get(int(year), {}).get(str(race))


def source_url(race, category, year=2026):
    if int(year) not in LM_EDITION or str(race) not in RACES:
        raise ValueError(f'Unrecognized year or race: {year} {race}')
    if category not in RACES[str(race)]['disc'].values():
        raise ValueError(f'Invalid category for {race} km: {category}')
    return 'https://ljubljanskimaraton.si/rezultati/posamezni?' + urlencode({
        'cat': category, 'lm': LM_EDITION[int(year)]})


def results_stem(race, year=2026):
    return f'{int(year)}_ljubljana_{race}k_results'
