import pandas as pd
import numpy as np

# ===========================================================
# Labels
# ===========================================================

LABELS = ['$t_0 \\, \\left( JD \\right)$',
          '$P \\, \\left( JD \\right)$',
          '$R_p \\left( \\frac{R_p}{R_s} \\right)$', 
          '$a \\left( R_s \\right)$',
          '$I \\left( \\text{degree} \\right)$', 
          'u1', 
          'u2'
]



# ===========================================================
# Finding Bounds
# ===========================================================

NEA = pd.read_csv(r"C:\Users\simon\OneDrive\Bureau\Exoplot_final\data\NEA.csv", comment="#")


def max_err(err1, err2):
    return max(err1, abs(err2))

def find_bounds(pl_name, scale_factor, df=NEA):

    mask = df['pl_name'].str.strip().str.lower() == pl_name.strip().lower()
    planet_row = df[mask].iloc[0]

    # orbital period
    period = planet_row['pl_orbper']
    period_err = max_err(planet_row['pl_orbpererr1'], planet_row['pl_orbpererr2'])

    # planet radius
    planet_radius = planet_row['pl_radj']
    planet_radius_err = max_err(planet_row['pl_radjerr1'], planet_row['pl_radjerr2'])

    # stellar radius
    stellar_radius = planet_row['st_rad']
    stellar_radius_err = max_err(planet_row['st_raderr1'], planet_row['st_raderr2'])

    # semi-major axis
    semi_major_axis = planet_row['pl_orbsmax']
    semi_major_axis_err = max_err(planet_row['pl_orbsmaxerr1'], planet_row['pl_orbsmaxerr2'])

    # inclination
    inclination = planet_row['pl_orbincl']
    inclination_err= max_err(planet_row['pl_orbinclerr1'], planet_row['pl_orbinclerr2'])

    # Rp/Rs
    RJ_RS = 71_492 / 696_340
    rp_rs = (planet_radius * RJ_RS) / stellar_radius
    rel_err_planet = planet_radius_err / planet_radius if planet_radius != 0 else 0
    rel_err_star = stellar_radius_err / stellar_radius if stellar_radius != 0 else 0
    rp_rs_err = rp_rs * ((rel_err_planet**2 + rel_err_star**2)**0.5)

    # a/Rs
    AU_Rsun = 215.032
    a_rs = (semi_major_axis / stellar_radius) * AU_Rsun
    rel_err_a = semi_major_axis_err / semi_major_axis if semi_major_axis != 0 else 0
    a_rs_err = a_rs * ((rel_err_a**2 + rel_err_star**2)**0.5)

    BOUNDS = [
        (0 - 0.3, 0 + 0.3),
        (period - 0.005, period + 0.005),
        (rp_rs - scale_factor * rp_rs_err, rp_rs + scale_factor * rp_rs_err),
        (a_rs - scale_factor * a_rs_err, a_rs + scale_factor * a_rs_err),
        (inclination - 2, inclination + 2 ),
        (0, 1),
        (0, 1)
    ]

    return BOUNDS


def compute_bounds(time, flux, flux_err, max_per):    
    BOUNDS = []
    
    # t0
    BOUNDS.append((np.min(time), np.max(time)))
    
    # P
    BOUNDS.append((max_per * 0.99, max_per * 1.01))
    
    # rp/rs
    rp_rs_est = np.sqrt(1 - np.min(flux)) - np.sqrt(np.mean(flux_err))
    x = 0.4 * rp_rs_est
    BOUNDS.append((max(rp_rs_est - x, 0.008), rp_rs_est + x))
    
    # a/rs
    a_rs_est = 2.8 * max_per**(2/3)
    y = 0.4 * a_rs_est
    BOUNDS.append((max(a_rs_est - y, 1.5), a_rs_est + y))
    
    # inclination
    BOUNDS.append((87, 93))
    
    # u1
    BOUNDS.append((0.0, 1.0))
    
    # u2
    BOUNDS.append((0.0, 1.0))
    
    return BOUNDS