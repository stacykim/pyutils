import tangos
import pynbody
import numpy as np

path = '/vol/ph/astro_data/shared/morkney/EDGE'

def Tangos_load(sim, redshift):

  global path

  print('Loading %s at z=%.2f' % (sim, redshift))

  tangos.core.init_db('{}/tangos/{}.db'.format(path, sim.split('_')[0]))
  session = tangos.core.get_default_session()

  # Find the output closest to this redshift:
  times = tangos.get_simulation('%s' % sim, session).timesteps

  final_time = times[-1].__dict__['extension']
  h = tangos.get_halo('{}/{}/halo_{:d}'.format(sim, final_time, 1), session)
  halo, z = h.calculate_for_progenitors('halo_number()', 'z()')

  z_ = np.array([times[i].__dict__['redshift'] for i in range(len(times))])[-len(halo):]
  output = np.array([times[i].__dict__['extension'] for i in range(len(times))])[-len(halo):]
  output = output[abs(z_ - redshift).argmin()]
  redshift = z_[abs(z_ - redshift).argmin()]

  # Find the progenitor halo number closest to this redshift:
  halo = halo[abs(z - redshift).argmin()]

  h = tangos.get_halo('{}/{}/halo_{:d}'.format(sim, output, halo))

  return h, session, output, halo, redshift




def rebin(x, y, new_x_bins, operation='mean'):

    index_radii = np.digitize(x, new_x_bins)
    if operation is 'mean':
        new_y = np.array([y[index_radii == i].mean() for i in range(1, len(new_x_bins))])
    elif operation is 'sum':
        new_y = np.array([y[index_radii == i].sum() for i in range(1, len(new_x_bins))])
    else:
        raise ValueError("Operation not supported")

    new_x = (new_x_bins[1:] + new_x_bins[:-1]) / 2
    new_y[np.isnan(new_y)] = np.finfo(float).eps

    return new_x, new_y
