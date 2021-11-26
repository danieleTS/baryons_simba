import numpy as np
import h5py

def get_cb_colours(name):
    # paul tol colour palette:
    # https://davidmathlogic.com/colorblind/#%23332288-%23117733-%2344AA99-%2388CCEE-%23DDCC77-%23CC6677-%23AA4499
    if name == 'tol':
        return ['grey', '#332288', '#117733', '#44AA99', '#88CCEE', '#DDCC77', '#CC6677', '#AA4499']
    # Okabe and Ito 2008
    # https://serialmentor.com/dataviz/color-pitfalls.html
    elif name == 'okabe':
        #return ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
        return ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00']

def get_bin_edges(x_min, x_max, dx):
    return np.arange(x_min, x_max+dx, dx)

def get_bin_middle(xbins):
    return np.array([xbins[i] + 0.5*(xbins[i+1] - xbins[i]) for i in range(len(xbins)-1)])

def bin_data(x, y, xbins, find_higher=False):
    digitized = np.digitize(x, xbins)
    binned_data = [y[digitized == i] for i in range(1, len(xbins))]
    if find_higher:
        ngals = [len(j) for j in binned_data]
        extra_bin = y[x > xbins[-1]]
        binned_data.append(extra_bin)
    return np.array(binned_data)

def group_high_mass(binned_data, min_ngals=10):
    j = len(binned_data) - 1
    ngals_end = len(binned_data[j])
    while ngals_end < min_ngals:
        binned_data[j-1] = np.concatenate((binned_data[j-1], binned_data[j]))
        binned_data[j] = np.array([])
        j -= 1
        ngals_end = len(binned_data[j])
    return binned_data

def convert_to_log(y, yerr):
    yerr /= (y*np.log(10.))
    y = np.log10(y)
    return y, yerr

def read_phases(phase_file, phases):
    phase_dict = {}
    with h5py.File(phase_file, 'r') as pf:
        for phase in phases:
            phase_dict[phase] = pf[phase][:]
    return phase_dict

def get_phase_stats(mass, pos, mass_budget, mask, phases, mass_bins, boxsize, logresults=False, min_ngals=10):
    
    stat_dict = {phase: {} for phase in phases}
    
    binned_pos = bin_data(mass[mask], pos[mask], 10.**mass_bins, find_higher=True)
    stat_dict['ngals'] = [len(j) for j in binned_pos]
    
    for phase in phases:
        binned_data = bin_data(mass[mask], mass_budget[phase][mask], 10.**mass_bins, find_higher=True)
        
        medians = np.zeros(len(mass_bins))
        cosmic_stds = np.zeros(len(mass_bins))
        for i in range(len(mass_bins)):
            if len(binned_data[i]) > 0.:
                #medians[i], cosmic_stds[i] = get_cosmic_variance_2d(binned_data[i], binned_pos[i], boxsize) # This iswhat Sarah used
                medians[i], cosmic_stds[i] = get_cosmic_variance(binned_data[i], binned_pos[i], boxsize)
            else:
                medians[i], cosmic_stds[i] = np.nan, np.nan
        if logresults:
            stat_dict[phase]['cosmic_median'], stat_dict[phase]['cosmic_std'] = convert_to_log(medians, cosmic_stds)
        else:
            stat_dict[phase]['cosmic_median'], stat_dict[phase]['cosmic_std'] = medians, cosmic_stds
        
        std = np.array([np.std(j) for j in binned_data])
        medians = np.array([np.nanpercentile(j, 50.) for j in binned_data])
        per25 = np.array([np.nanpercentile(j, 25.) for j in binned_data])
        per75 = np.array([np.nanpercentile(j, 75.) for j in binned_data])
        upper = per75 - medians
        lower = medians - per25
        if logresults:
            stat_dict[phase]['median'], stat_dict[phase]['percentile_25_75'] = convert_to_log(medians, np.array([lower, upper]))
            _, stat_dict[phase]['std'] = convert_to_log(medians, std)
        else:
            stat_dict[phase]['median'], stat_dict[phase]['percentile_25_75'] = medians, np.array([lower, upper])
            stat_dict[phase]['std'] = std

    return stat_dict

def read_phase_stats(stats_file, phases, stats):
    stats_dict = {}
    with h5py.File(stats_file, 'r') as sf:
        for cut in ['all']:#, 'star_forming', 'quenched']:
            stats_dict[cut] = {p: {} for p in phases}
            stats_dict[cut]['ngals'] = sf[cut]['ngals'][:]
            for phase in phases:
                for stat in stats:
                    stats_dict[cut][phase][stat] = sf[cut][phase][stat][:]
        
    stats_dict['hmass_bins'] = sf['hmass_bins'][:]
    return stats_dict

def write_phase_stats(stats_file, stats_dict, phases, stats, ):
    with h5py.File(stats_file, 'a') as hf:
        for cut in ['all']:#, 'star_forming', 'quenched']:
            cut_grp = hf.create_group(cut)
            cut_grp.create_dataset('ngals', data=np.array(stats_dict[cut]['ngals']))
            for phase in phases:
                phase_grp = cut_grp.create_group(phase)
                for stat in stats:
                    phase_grp.create_dataset(stat, data=np.array(stats_dict[cut][phase][stat]))
        hf.create_dataset('hmass_bins', data=np.array(stats_dict['hmass_bins']))
    return

def variance_jk(samples, mean):
    n = len(samples)
    factor = (n-1.)/n
    x = np.nansum((np.subtract(samples, mean))**2, axis=0)
    x *= factor
    return x

def get_cosmic_variance_2d(quantity, pos, boxsize):
    octant_ids = octants_2d(pos, boxsize)
    measure = np.zeros(8)
    for i in range(8):
        i_using = np.concatenate(np.delete(octant_ids, i))
        measure[i] = np.nanmedian(quantity[i_using.astype('int')])
    mean_m = np.nansum(measure) / 8.
    cosmic_std = np.sqrt(variance_jk(measure, mean_m))
    return mean_m, cosmic_std

def get_cosmic_variance(quantity, pos, boxsize, per=False):
    octant_ids = octants_3d(pos, boxsize)
    measure = np.zeros(8)
    for i in range(8):
        i_using = np.concatenate(np.delete(octant_ids, i))
        measure[i] = np.nanmedian(quantity[i_using.astype('int')])
    mean_m = np.nansum(measure) / 8.
    cosmic_std = np.sqrt(variance_jk(measure, mean_m))
    if (per):
        per_inf = np.nanpercentile(measure, 16)
        per_sup = np.nanpercentile(measure, 84)
        return mean_m, per_inf, per_sup
    else:
        return mean_m, cosmic_std

def N_scatter(quantity, pos, boxsize, per=False):
    octant_ids = octants_3d(pos, boxsize)
    measure = np.zeros(8)
    for i in range(8):
        i_using = np.concatenate(np.delete(octant_ids, i))
        measure[i] = len(quantity[i_using.astype('int')])
    mean_m = np.nansum(measure) / 8.
    cosmic_std = np.sqrt(variance_jk(measure, mean_m))
    if (per):
        per_inf = np.nanpercentile(measure, 16)
        per_sup = np.nanpercentile(measure, 84)
        return mean_m, per_inf, per_sup
    else:
        return mean_m, cosmic_std
    
def octants_3d(pos_array, boxsize):
    pos_x = (pos_array[:, 0] < boxsize*0.5)
    pos_y = (pos_array[:, 1] < boxsize*0.5)
    pos_z = (pos_array[:, 2] < boxsize*0.5)
    
    inds_1 = np.array([]); inds_2 = np.array([]); inds_3 = np.array([]); inds_4 = np.array([])
    inds_5 = np.array([]); inds_6 = np.array([]); inds_7 = np.array([]); inds_8 = np.array([])
    
    for i in range(len(pos_array)):
        if (pos_x[i] and (pos_y[i] and pos_z[i])):
            inds_1 = np.append(inds_1, i)
        elif (pos_x[i] and (pos_y[i] and not pos_z[i])):
            inds_2 = np.append(inds_2, i)
        elif (pos_x[i] and (pos_z[i] and not pos_y[i])):
            inds_3 = np.append(inds_3, i)
        elif (pos_x[i] and not (pos_y[i] or pos_z[i])):
            inds_4 = np.append(inds_4, i)
        elif ((pos_y[i] and pos_z[i]) and not pos_x[i]):
            inds_5 = np.append(inds_5, i)
        elif ((pos_y[i] and not pos_z[i]) and not pos_x[i]):
            inds_6 = np.append(inds_6, i)
        elif ((pos_z[i] and not pos_y[i]) and not pos_x[i]):
            inds_7 = np.append(inds_7, i)
        elif (not(pos_x[i]) and not (pos_y[i] or pos_z[i])):
            inds_8 = np.append(inds_8, i)

    return np.array((inds_1, inds_2, inds_3, inds_4, inds_5, inds_6, inds_7, inds_8))

def octants_2d(pos_array, boxsize):
    pos_x = (pos_array[:, 0] < boxsize*0.5)
    pos_ya = (pos_array[:, 1] < boxsize*0.25)
    pos_yb = (pos_array[:, 1] > boxsize*0.25)& (pos_array[:, 1] < boxsize*0.5)
    pos_yc = (pos_array[:, 1] > boxsize*0.5)& (pos_array[:, 1] < boxsize*0.75)
    pos_yd = (pos_array[:, 1] > boxsize*0.75)
    inds_1 = np.array([]); inds_2 = np.array([]); inds_3 = np.array([]); inds_4 = np.array([])
    inds_5 = np.array([]); inds_6 = np.array([]); inds_7 = np.array([]); inds_8 = np.array([])
    
    for i in range(len(pos_array)):
        if pos_ya[i] and pos_x[i]:
            inds_1 = np.append(inds_1, i)
        elif pos_ya[i] and not pos_x[i]:
            inds_2 = np.append(inds_2, i)
        elif pos_yb[i] and pos_x[i]:
            inds_3 = np.append(inds_3, i)
        elif pos_yb[i] and not pos_x[i]:
            inds_4 = np.append(inds_4, i)
        elif pos_yc[i] and pos_x[i]:
            inds_5 = np.append(inds_5, i)
        elif pos_yc[i] and not pos_x[i]:
            inds_6 = np.append(inds_6, i)
        elif pos_yd[i] and pos_x[i]:
            inds_7 = np.append(inds_7, i)
        elif pos_yd[i] and not pos_x[i]:
            inds_8 = np.append(inds_8, i)

    return np.array((inds_1, inds_2, inds_3, inds_4, inds_5, inds_6, inds_7, inds_8))

