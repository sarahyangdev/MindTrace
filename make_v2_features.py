"""Feature engineering: add spatial-context, density and relative-deprivation features."""
import pandas as pd, numpy as np
from scipy.spatial import cKDTree

SCR = r'C:\Users\bette\AppData\Local\Temp\claude\C--MailEasy-CRMAI-crmai-leads-views\a8979f87-282a-4588-b2f4-b89eacb9cde6\scratchpad'
m = pd.read_csv(r'C:\Users\bette\Downloads\mindtrace_master_dataset_ca_tracts.csv',
                dtype={'tract_fips': str, 'county_fips': str, 'state_fips': str})
n0 = m.shape[1]

# ---- land area + population density (from the 2020 tract crosswalk) ----
xw = pd.read_csv(f'{SCR}\\xwalk.txt', sep='|', dtype=str,
                 usecols=['GEOID_TRACT_20', 'AREALAND_TRACT_20'])
xw = xw[xw.GEOID_TRACT_20.str.startswith('06')].drop_duplicates('GEOID_TRACT_20')
xw['land_sqmi'] = pd.to_numeric(xw.AREALAND_TRACT_20) / 2_589_988.0
m = m.merge(xw[['GEOID_TRACT_20', 'land_sqmi']].rename(columns={'GEOID_TRACT_20': 'tract_fips'}),
            on='tract_fips', how='left')
m['pop_density_per_sqmi'] = m.total_population / m.land_sqmi.replace(0, np.nan)
m['log_pop_density'] = np.log1p(m.pop_density_per_sqmi)
m['pct_adult'] = 100 * m.total_pop_18plus / m.total_population.replace(0, np.nan)

# ---- spatial lag: mean of the 10 nearest tracts (excluding self) ----
R = 3958.7614
la, lo = np.radians(m.tract_lat.values), np.radians(m.tract_lon.values)
pts = np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)] * R
tree = cKDTree(pts)
_, idx = tree.query(pts, k=11)          # col 0 is the tract itself
nb = idx[:, 1:]

LAG = ['poverty_rate', 'housing_burden_pct', 'education_pct_less_hs', 'unemployment_rate',
       'linguistic_isolation_pct', 'pm25_annual', 'ozone_mean_ppb_2yr', 'tree_canopy_pct',
       'log_pop_density', 'dist_to_mental_health']
for c in LAG:
    v = m[c].values
    m[f'nb10_{c}'] = np.nanmean(v[nb], axis=1)

# ---- relative deprivation: tract vs its own county and vs its neighbours ----
for c in ['poverty_rate', 'housing_burden_pct', 'education_pct_less_hs', 'unemployment_rate',
          'pm25_annual', 'dist_to_mental_health']:
    cty = m.groupby('county_fips')[c].transform('mean')
    m[f'{c}_vs_county'] = m[c] - cty
    m[f'{c}_county_mean'] = cty
m['poverty_vs_neighbours'] = m.poverty_rate - m.nb10_poverty_rate
m['housing_burden_vs_neighbours'] = m.housing_burden_pct - m.nb10_housing_burden_pct

# ---- gravity access score: every facility discounted by distance ----
f = pd.read_csv(r'C:\Users\bette\Downloads\health_facility_locations.csv', low_memory=False)
f = f[f.LATITUDE.notna() & f.LONGITUDE.notna()]
MH = {'ACUTE PSYCHIATRIC HOSPITAL', 'PSYCHOLOGY CLINIC',
      'CHEMICAL DEPENDENCY RECOVERY HOSPITAL', 'CORRECTIONAL TREATMENT CENTER'}
for name, sub in [('mh', f[f.FAC_FDR.isin(MH)]), ('pcp', f[f.FAC_FDR == 'PRIMARY CARE CLINIC'])]:
    fla, flo = np.radians(sub.LATITUDE.values), np.radians(sub.LONGITUDE.values)
    fp = np.c_[np.cos(fla) * np.cos(flo), np.cos(fla) * np.sin(flo), np.sin(fla)] * R
    ft = cKDTree(fp)
    k = min(50, len(fp))
    d, _ = ft.query(pts, k=k)
    gc = 2 * R * np.arcsin(np.clip(d / (2 * R), 0, 1))
    m[f'gravity_access_{name}'] = (1.0 / (1.0 + gc ** 2)).sum(axis=1)
m['mh_facilities_per_100k'] = 1e5 * m.n_mental_health_within_25mi / \
    m.groupby('county_fips').total_population.transform('sum')

# ---- interactions between the strongest drivers ----
m['poverty_x_housing_burden'] = m.poverty_rate * m.housing_burden_pct / 100
m['poverty_x_pm25'] = m.poverty_rate * m.pm25_annual / 100
m['poverty_x_isolation'] = m.poverty_rate * m.linguistic_isolation_pct / 100
m['deprivation_index'] = m[['poverty_rate', 'housing_burden_pct', 'education_pct_less_hs',
                            'unemployment_rate', 'linguistic_isolation_pct']].rank(pct=True).mean(axis=1) * 100
m['pollution_x_deprivation'] = m.deprivation_index * m.pm25_annual / 100

# ---- air-quality shape ----
m['pm25_peak_ratio'] = m.pm25_max_daily_2021 / m.pm25_annual.replace(0, np.nan)
m['aq_combined_burden'] = (m[['pm25_annual', 'ozone_mean_ppb_2yr', 'ces4_diesel_pm']]
                           .rank(pct=True).mean(axis=1) * 100)

m = m.replace([np.inf, -np.inf], np.nan)
new = [c for c in m.columns if c not in pd.read_csv(
    r'C:\Users\bette\Downloads\mindtrace_master_dataset_ca_tracts.csv', nrows=1).columns]
m.to_csv(r'C:\Users\bette\Downloads\mindtrace_master_dataset_ca_tracts_v2.csv', index=False)
print(f'v2 written: {m.shape[0]} rows x {m.shape[1]} cols  (+{m.shape[1]-n0} engineered)')
print('new features:'); [print('  ', c) for c in new]
