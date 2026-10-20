"""Single source of truth for the 2026 official Ljubljana Marathon distances."""
from urllib.parse import urlencode

EVENT_YEAR = 2026
RACES = {
    '10': {'label': '10 km · Garmin', 'long_name': 'Garmin 10 km', 'km': 10.0,
           'event': 7091, 'disc': {'M': '10M', 'F': '10Z'}, 'checkpoints': [5],
           'age_disc': {}, 'date': '2026-10-17'},
    '21': {'label': '21.1 km · Half marathon', 'long_name': 'Half marathon', 'km': 21.0975,
           'event': 7092, 'disc': {'M': '21M', 'F': '21Z'}, 'checkpoints': [5, 10, 15, 20],
           'age_disc': {'M': [f'21M{a}' for a in 'ABCDEFGHI'],
                        'F': [f'21Z{a}' for a in 'ABCDEFGHI']}, 'date': '2026-10-18'},
    '42': {'label': '42.2 km · Marathon', 'long_name': 'Marathon', 'km': 42.195,
           'event': 7092, 'disc': {'M': '42M', 'F': '42Z'}, 'checkpoints': [10, 20, 30],
           'age_disc': {'M': [f'42M{a}' for a in 'ABCDEFGHI'],
                        'F': [f'42Z{a}' for a in 'ABCDEFGHI']}, 'date': '2026-10-18'},
}
ALL_SPLITS = [5, 10, 15, 20, 30]


def source_url(race, category):
    config = RACES[str(race)]
    return 'https://www.timingljubljana.si/rezultati.aspx?' + urlencode({
        'disc': category, 'embed': '', 'idTekme': config['event'], 'tip': 'B'})


def results_stem(race):
    return f'2026_ljubljana_{race}k_results'
