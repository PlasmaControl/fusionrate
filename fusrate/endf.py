import numba
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

import fusrate.reactionnames as rn
from fusrate.ion_data import ion_mass
from fusrate.load_data import cross_section_data


class LogLogExtrapolation:
    r"""Interpolate and extrapolate in log-log space

    with a straight-line right end in log-log space
    """
    SMALL = 1e-50  # to prevent errors with log of 0

    def __init__(self, x, y, linear_extension=True):
        r"""
        x: array_like
        y: array_like
        """
        self.x = x
        self.y = y
        self.max_x = max(x)
        self.min_x = min(x)
        logx = np.log(x + self.SMALL)
        logy = np.log(y)
        data = np.array([logx, logy]).T

        # linear extension in loglog space
        if linear_extension:
            last_two = data[-2:]
            last = last_two[-1]
            Δ = last_two[-1] - last_two[-2]
            subsequent_points = last + np.outer([1, 2, 3], Δ)
            self.data = np.append(data, subsequent_points, axis=0)
        else:
            self.data = data

        self.logx, self.logy = self.data.T

        self.interpolator = InterpolatedUnivariateSpline(
            self.logx, self.logy, k=2, ext=0
        )

    def __call__(self, newx):
        r"""Generate new values

        Parameters
        ----------
        newx: array_like
            Energies in eV.
            Whether this is beam-target energy or c.o.m. energy
            is up to the data.
        """
        log_new = np.log(newx + self.SMALL)
        log_newy = self.interpolator(log_new)
        return np.exp(log_newy)


class LogLogReinterpolation:
    r"""Re-sampled LogLogExtrapolation

    This creates an interpolation function with a consistent spacing (in log-x
    space), which allows using the np.interp function. The latter is useful
    because it can be 'jit-compiled' using numba.
    """
    SMALL = 1e-50  # barns
    LOWBOUND = 0.010  # keV
    HIGHBOUND = 4e4  # keV
    NUM_REMESH = 6000

    def __init__(self, x, y, linear_extension=True, num_remesh=None):
        r"""
        x: array_like
        y: array_like
        """
        if num_remesh is not None:
            self.NUM_REMESH = num_remesh
        self.remeshed_logx = self._remeshed_log_x()

        lle = LogLogExtrapolation(x, y, linear_extension)
        self.remeshed_logy = lle.interpolator(self.remeshed_logx)

    def _remeshed_log_x(self):
        return np.linspace(
            np.log(self.LOWBOUND), np.log(self.HIGHBOUND), self.NUM_REMESH
        )

    def __call__(self, newx):
        r"""Generate new values
        newx: array_like,
            energies in eV.
            Whether this is beam-target energy or c.o.m. energy
            is up to the data.
        """
        log_new = np.log(newx)
        log_newy = np.interp(
            log_new,
            self.remeshed_logx,
            self.remeshed_logy,
            right=np.log10(self.SMALL),
        )
        return np.exp(log_newy)

    def make_jitfunction(self):
        small = self.SMALL
        remesh_logx = self.remeshed_logx
        remesh_logy = self.remeshed_logy

        @numba.njit(cache=True, fastmath=True)
        def logloginterp(newx):
            log_new = np.log(newx + small)
            log_newy = np.interp(log_new, remesh_logx, remesh_logy)
            return np.exp(log_newy)

        return logloginterp


class ENDFCrossSection:
    def __init__(self, r, interpolation="LogLogExtrapolation"):
        r"""
        s: reaction name string
        """
        if isinstance(r, str):
            name = rn.name_resolver(r)

            beam, target = rn.reactants(name)

            m_beam = ion_mass(beam)
            m_tar = ion_mass(target)

            self.bt_to_com = m_tar / (m_beam + m_tar)
        else:
            name = r.name
            self.bt_to_com = r.bt_to_com

        self.canonical_reaction_name = name
        x_raw, y_raw = cross_section_data(name)

        # Change from lab frame to COM frame
        # and from eV to keV (to match typical scales and Bosch-Hale)
        x = x_raw * self.bt_to_com / 1e3

        # Change from b to mb
        y = y_raw * 1e3

        self.x = x

        if interpolation == "LogLogExtrapolation":
            self.interp = LogLogExtrapolation(x, y, linear_extension=True)
        elif interpolation == "LogLogReinterpolation":
            interp_source = LogLogReinterpolation(x, y, linear_extension=True)
            self.interp = interp_source.make_jitfunction()
        else:
            raise ValueError(
                f"Unknown interpolation type {interpolation}."
                "Allowed values are LogLogExtrapolation and"
                "LogLogReinterpolation."
            )

    def __call__(self, e):
        return self.cross_section(e)

    def cross_section(self, e):
        r"""Look up the cross section from ENDF data
        Parameters
        ----------
        e : array_like,
          energies in keV

        Returns
        -------
        Cross sections in mb
        """
        return self.interp(e)

    def prescribed_range(self):
        r"""
        Returns
        -------
        [min, max] of COM energy range in keV
        """
        return [min(self.x), max(self.x)]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from fusrate.load_data import load_data_file

    endf = ENDFCrossSection("D+T")
    llr = ENDFCrossSection("D+T", interpolation="LogLogReinterpolation")

    newx = np.logspace(0, 3, 100)

    print(llr.prescribed_range())

    plt.loglog(newx, endf(newx))
    plt.loglog(newx, llr(newx))
    plt.show()
