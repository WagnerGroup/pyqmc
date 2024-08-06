"""
    adapted from
    https://github.com/lab-cosmo/sphericart/blob/main/sphericart/include/macros.hpp
    https://doi.org/10.1063/5.0156307
    Filippo Bigi  Guillaume Fraux  Nicholas J. Browning Michele Ceriotti 
    J. Chem. Phys. 159, 064802 (2023)

    Hard-coded expressions for low-l Cartesian spherical harmonics and their
   derivatives.

    These are implemented as macros to be included anywhere at compile time.
   They assume values are computed for one point at a time, that the
   spherical harmonics are stored in a contiguous section that "flattens" the
   (l,m) dimensions, e.g. [ (0,0), (1,-1), (1,0), (1,1), (2,-2), ...]

    Functions get pointers to the beginning of the storage space for the
   current sample, x,y, and, l>1, x^2, y^2 and z^2, can be reused.

    Each macro computes one l, macros should be called in order as the
   higher l reuse calculations at lower angular momentum. The expressions here
   are derived with computer assisted algebra by attempting all possible
   polynomial decompositions and selecting that with the smallest number of
   operations. One should call COMPUTE_SPH_L* or COMPUTE_SPH_DERIVATIVE_L*
   depending on whether only Ylm are needed or if one also want to evbaluate
   Cartesian derivatives

    Every macro takes an agument  that is an indexing function, can
   be used to map the consecutive indices of the Ylm to a different memory
   layout (self is e.g. used to optimize threads in CUDA code)
"""

from numba import njit

# self is used thoughout to indicate the maximum l channel for which we
# provide a hard-coded macro. self should be modified if further macros are
# added
SPHERICART_LMAX_HARDCODED = 6


@njit(fastmath=True, cache=True)
def COMPUTE_SPH_L0(
    sph_i,
):
    sph_i[(0)] = 0.282094791773878


@njit(fastmath=True, cache=True)
def COMPUTE_SPH_DERIVATIVE_L0(
    sph_i,
    dx_sph_i,
    dy_sph_i,
    dz_sph_i,
):
    dx_sph_i[(0)] = dy_sph_i[(0)] = dz_sph_i[(0)] = 0.0


@njit(fastmath=True, cache=True)
def COMPUTE_SPH_L1( x, y, z, sph_i,):
    sph_i[(1)] = (0.48860251190292) * x
    sph_i[(2)] = (0.48860251190292) * y
    sph_i[(3)] = (0.48860251190292) * z


@njit(fastmath=True, cache=True)
def COMPUTE_SPH_DERIVATIVE_L1( sph_i, dx_sph_i, dy_sph_i, dz_sph_i,):
    dx_sph_i[(1)] = 0.48860251190292
    dx_sph_i[(2)] = 0.0
    dx_sph_i[(3)] = 0.0
    dy_sph_i[(1)] = 0.0
    dy_sph_i[(2)] = 0.48860251190292
    dy_sph_i[(3)] = 0.0
    dz_sph_i[(1)] = 0.0
    dz_sph_i[(2)] = 0.0
    dz_sph_i[(3)] = 0.48860251190292


@njit(fastmath=True, cache=True)
def COMPUTE_SPH_L2( x, y, z, x2, y2, z2, sph_i,):
    sph_i[(4)] = 1.0925484305920792 * x * y
    sph_i[(7)] = 1.0925484305920792 * x * z
    sph_i[(5)] = 1.0925484305920792 * y * z
    sph_i[(6)] = -(0.315391565252520) * (x2 + y2 - 2 * z2)
    sph_i[(8)] = (0.54627421529604) * (x2 - y2)


@njit(fastmath=True, cache=True)
def COMPUTE_SPH_DERIVATIVE_L2(
    x, y, z, x2, y2, z2, sph_i, dx_sph_i, dy_sph_i, dz_sph_i,
):
    dx_sph_i[(4)] = (1.0925484305920792) * y
    dx_sph_i[(5)] = 0.0
    dx_sph_i[(6)] = -(0.6307831305050422) * x
    dx_sph_i[(7)] = (1.0925484305920792) * z
    dx_sph_i[(8)] = (1.0925484305920792) * x

    dy_sph_i[(4)] = dx_sph_i[(8)]
    dy_sph_i[(5)] = dx_sph_i[(7)]
    dy_sph_i[(6)] = -(0.6307831305050433) * y
    dy_sph_i[(7)] = 0.0
    dy_sph_i[(8)] = -dx_sph_i[(4)]

    dz_sph_i[(4)] = 0.0
    dz_sph_i[(5)] = dx_sph_i[(4)]
    dz_sph_i[(6)] = 1.2615662610100846 * z
    dz_sph_i[(7)] = dy_sph_i[(4)]
    dz_sph_i[(8)] = 0.0


@njit(fastmath=True, cache=True)
def COMPUTE_SPH_L3( x, y, z, x2, y2, z2, sph_i,):
    sph_i[(9)] = -(0.59004358992664) * y * (y2 - 3 * x2)
    sph_i[(10)] = (2.64575131106459) * z * sph_i[(4)]
    tmp = -(0.457045799464466) * (x2 + y2 - 4 * z2)
    sph_i[(11)] = y * tmp
    sph_i[(13)] = x * tmp
    sph_i[(12)] = -(1.49270533036046) * z * (z2 - (2.37799637856361) * sph_i[(6)])
    sph_i[(14)] = (1.44530572132028) * z * (x2 - y2)
    sph_i[(15)] = (0.59004358992664) * x * (x2 - 3 * y2)


@njit(fastmath=True, cache=True)
def COMPUTE_SPH_DERIVATIVE_L3(
    x, y, z, x2, y2, z2, sph_i, dx_sph_i, dy_sph_i, dz_sph_i,
):
    dx_sph_i[(9)] = (3.24037034920393) * sph_i[(4)]
    dx_sph_i[(10)] = (2.64575131106459) * sph_i[(5)]
    dx_sph_i[(11)] = -(0.83666002653408) * sph_i[(4)]
    dx_sph_i[(12)] = -(2.04939015319192) * sph_i[(7)]
    dx_sph_i[(13)] = (0.8366600265340735) * (-sph_i[(8)] + (3.464101615137758) * sph_i[(6)])
    dx_sph_i[(14)] = (2.64575131106459) * sph_i[(7)]
    dx_sph_i[(15)] = (3.24037034920393) * sph_i[(8)]

    dy_sph_i[(9)] = dx_sph_i[(15)]
    dy_sph_i[(10)] = dx_sph_i[(14)]
    dy_sph_i[(11)] = (0.8366600265340735) * (sph_i[(8)] + (3.464101615137758) * sph_i[(6)])
    dy_sph_i[(12)] = -(2.04939015319192) * sph_i[(5)]
    dy_sph_i[(13)] = -(0.83666002653408) * sph_i[(4)]
    dy_sph_i[(14)] = -dx_sph_i[(10)]
    dy_sph_i[(15)] = -dx_sph_i[(9)]

    dz_sph_i[(9)] = 0.0
    dz_sph_i[(10)] = (2.64575131106459) * sph_i[(4)]
    dz_sph_i[(11)] = (3.34664010613630) * sph_i[(5)]
    dz_sph_i[(12)] = (3.54964786985977) * sph_i[(6)]
    dz_sph_i[(13)] = (3.34664010613630) * sph_i[(7)]
    dz_sph_i[(14)] = (2.64575131106459) * sph_i[(8)]
    dz_sph_i[(15)] = 0.0


@njit(fastmath=True, cache=True)
def COMPUTE_SPH_L4( x, y, z, x2, y2, z2, sph_i,):
    sph_i[(16)] = (4.194391357527674) * sph_i[(4)] * sph_i[(8)]
    sph_i[(17)] = 3 * z * sph_i[(9)]
    tmp = -(0.866025403784439) * (x2 + y2 - 6 * z2)
    sph_i[(18)] = tmp * sph_i[(4)]
    sph_i[(22)] = tmp * sph_i[(8)]
    sph_i[(20)] = -(0.69436507482941) * (
        y * sph_i[(11)] - (1.6329931618554521) * z * sph_i[(12)] + x * sph_i[(13)]
    )
    tmp = -(1.224744871391589) * (z2 - (4.755992757127213) * sph_i[(6)])
    sph_i[(19)] = sph_i[(5)] * tmp
    sph_i[(21)] = sph_i[(7)] * tmp
    sph_i[(23)] = 3 * z * sph_i[(15)]
    sph_i[(24)] = -(1.060660171779821) * (y * sph_i[(9)] - x * sph_i[(15)])


@njit(fastmath=True, cache=True)
def COMPUTE_SPH_DERIVATIVE_L4(
    x, y, z, x2, y2, z2, sph_i, dx_sph_i, dy_sph_i, dz_sph_i,
):
    dx_sph_i[(16)] = (4.242640687119285) * sph_i[(9)]
    dx_sph_i[(17)] = (3.674234614174767) * sph_i[(10)]
    dx_sph_i[(18)] = (1.892349391515120) * y * (y2 + (4.755992757127213) * sph_i[(6)])
    dx_sph_i[(19)] = -(1.388730149658827) * sph_i[(10)]
    dx_sph_i[(20)] = -(2.777460299317654) * sph_i[(13)]
    dx_sph_i[(21)] = 2.676186174229157 * z * (z2 - 2.25 * x2 - .75 * y2)
    dx_sph_i[(22)] = -(1.892349391515120) * x * (x2 - 3 * z2)
    dx_sph_i[(23)] = (3.674234614174767) * sph_i[(14)]
    dx_sph_i[(24)] = (4.242640687119285) * sph_i[(15)]

    dy_sph_i[(16)] = dx_sph_i[(24)]
    dy_sph_i[(17)] = dx_sph_i[(23)]
    dy_sph_i[(18)] = (
        -(1.892349391515120) * x * (y2 - 2 * z2 - (1.585330919042404) * sph_i[(6)])
    )
    dy_sph_i[(19)] = 2.676186174229157 * z * (z2 - 0.75 * x2 - 2.25 * y2)
    dy_sph_i[(20)] = -(2.777460299317654) * sph_i[(11)]
    dy_sph_i[(21)] = dx_sph_i[(19)]
    dy_sph_i[(22)] = (1.892349391515120) * y * (y2 - 3 * z2)
    dy_sph_i[(23)] = -dx_sph_i[(17)]
    dy_sph_i[(24)] = -dx_sph_i[(16)]

    dz_sph_i[(16)] = 0.0
    dz_sph_i[(17)] = 3 * sph_i[(9)]
    dz_sph_i[(18)] = (3.927922024247863) * sph_i[(10)]
    dz_sph_i[(19)] = (4.391550328268399) * sph_i[(11)]
    dz_sph_i[(20)] = (4.535573676110727) * sph_i[(12)]
    dz_sph_i[(21)] = (4.391550328268399) * sph_i[(13)]
    dz_sph_i[(22)] = (3.927922024247863) * sph_i[(14)]
    dz_sph_i[(23)] = 3 * sph_i[(15)]
    dz_sph_i[(24)] = 0.0


@njit(fastmath=True, cache=True)
def COMPUTE_SPH_L5( x, y, z, x2, y2, z2, sph_i,):
    sph_i[(25)] = (
        (13.12764113680340)
        * y
        * (y2 * (x2 - (0.2) * y2) + (0.3994658435740642) * sph_i[(24)])
    )
    tmp = (3.316624790355400) * z
    sph_i[(26)] = tmp * sph_i[(16)]
    sph_i[(34)] = tmp * sph_i[(24)]
    tmp = (4.974937185533100) * (z2 + (0.5284436396808015) * sph_i[(6)])
    sph_i[(27)] = tmp * sph_i[(9)]
    sph_i[(33)] = tmp * sph_i[(15)]
    tmp = (5.257947827012948) * sph_i[(6)]
    sph_i[(28)] = tmp * sph_i[(10)]
    sph_i[(32)] = tmp * sph_i[(14)]
    tmp = (0.6324555320336759) * z
    sph_i[(29)] = (1.427248064296125) * (y * sph_i[(20)] + tmp * sph_i[(19)])
    sph_i[(31)] = (1.427248064296125) * (x * sph_i[(20)] + tmp * sph_i[(21)])
    sph_i[(30)] = (1.403403869441083) * (
        (3.540173863740353) * sph_i[(6)] * sph_i[(12)] - z * z2 * z2
    )
    sph_i[(35)] = -(1.048808848170152) * (y * sph_i[(16)] - x * sph_i[(24)])


@njit(fastmath=True, cache=True)
def COMPUTE_SPH_DERIVATIVE_L5(
    x, y, z, x2, y2, z2, sph_i, dx_sph_i, dy_sph_i, dz_sph_i,
):
    dx_sph_i[(25)] = (5.244044240850758) * sph_i[(16)]
    dx_sph_i[(26)] = (4.690415759823430) * sph_i[(17)]
    dx_sph_i[(27)] = (3.582364210034113) * (
        y2 * sph_i[(4)] + (3.58568582800318) * x * sph_i[(11)]
    )
    dx_sph_i[(28)] = -(8.774964387392122) * (
        (y2 - z2) * sph_i[(5)] + (0.3086066999241838) * sph_i[(17)]
    )
    dx_sph_i[(29)] = -(1.914854215512676) * sph_i[(18)]
    dx_sph_i[(30)] = -(3.496029493900505) * sph_i[(21)]
    dx_sph_i[(31)] = -(8.616843969807043) * (
        (0.2102610435016800) * z2 * z2
        + (1.056887279361603) * sph_i[(5)] * sph_i[(5)]
        + (y2 - z2) * sph_i[(6)]
        + (0.555555555555556) * sph_i[(22)]
    )
    dx_sph_i[(32)] = -(8.774964387392122) * (x2 - z2) * sph_i[(7)]
    dx_sph_i[(33)] = -(5.170697352496190) * (
        (0.106904496764970) * z * dx_sph_i[(23)]
        - (0.320713490294909) * y * sph_i[(9)]
        - sph_i[(22)]
    )
    dx_sph_i[(34)] = (4.690415759823430) * sph_i[(23)]
    dx_sph_i[(35)] = (5.24404424085076) * sph_i[(24)]

    dy_sph_i[(25)] = dx_sph_i[(35)]
    dy_sph_i[(26)] = dx_sph_i[(34)]
    dy_sph_i[(27)] = -(3.102418411497714) * (
        (0.534522483824849) * y * sph_i[(9)]
        - (0.654653670707977) * z * sph_i[(14)]
        - sph_i[(22)]
    )
    dy_sph_i[(28)] = (
        -(8.77496438739212) * (y2 - (1.585330919042404) * sph_i[(6)]) * sph_i[(7)]
    )
    dy_sph_i[(29)] = (0.7237468644557459) * (
        y * ((2.12132034355964) * sph_i[(9)] - (8.21583836257749) * sph_i[(11)])
        + (6.70820393249937) * z * sph_i[(12)]
        + sph_i[(24)]
    )
    dy_sph_i[(30)] = -(3.496029493900505) * sph_i[(19)]
    dy_sph_i[(31)] = dx_sph_i[(29)]
    dy_sph_i[(32)] = (8.77496438739212) * (y2 - z2) * sph_i[(5)]
    dy_sph_i[(33)] = (
        (3.582364210034113)
        * sph_i[(4)]
        * (y2 - 5 * z2 - (1.585330919042404) * sph_i[(6)])
    )
    dy_sph_i[(34)] = -dx_sph_i[(26)]
    dy_sph_i[(35)] = -dx_sph_i[(25)]

    dz_sph_i[(25)] = 0.0
    dz_sph_i[(26)] = (3.316624790355400) * sph_i[(16)]
    dz_sph_i[(27)] = (4.422166387140533) * sph_i[(17)]
    dz_sph_i[(28)] = (5.066228051190221) * sph_i[(18)]
    dz_sph_i[(29)] = (5.416025603090640) * sph_i[(19)]
    dz_sph_i[(30)] = (5.527707983925666) * sph_i[(20)]
    dz_sph_i[(31)] = (5.416025603090640) * sph_i[(21)]
    dz_sph_i[(32)] = (5.066228051190221) * sph_i[(22)]
    dz_sph_i[(33)] = (4.422166387140533) * sph_i[(23)]
    dz_sph_i[(34)] = (3.316624790355400) * sph_i[(24)]
    dz_sph_i[(35)] = 0.0


@njit(fastmath=True, cache=True)
def COMPUTE_SPH_L6( x, y, z, x2, y2, z2, sph_i,):
    sph_i[(36)] = (3.924637560539857) * sph_i[(9)] * sph_i[(15)]
    tmp = (3.605551275463989) * z
    sph_i[(37)] = tmp * sph_i[(25)]
    sph_i[(47)] = tmp * sph_i[(35)]
    tmp = (6.4498061986388) * (z2 + (0.396332729760601) * sph_i[(6)])
    sph_i[(38)] = tmp * sph_i[(16)]
    sph_i[(46)] = tmp * sph_i[(24)]
    tmp = (1.04083299973307) * (z2 + (4.75599275712721) * sph_i[(6)])
    sph_i[(39)] = tmp * sph_i[(17)]
    sph_i[(45)] = tmp * sph_i[(23)]
    sph_i[(40)] = (2.033805211017918) * (
        (0.3779644730092272) * z * sph_i[(28)] + x * sph_i[(29)]
    )
    tmp = -(6.399218702310463) * (
        z2 * z2 - (4.188790204786391) * sph_i[(6)] * sph_i[(6)]
    )
    sph_i[(41)] = tmp * sph_i[(5)]
    sph_i[(43)] = tmp * sph_i[(7)]
    sph_i[(42)] = -(1.087114613009218) * (
        (0.645497224367903) * y * sph_i[(29)]
        - z * sph_i[(30)]
        + (0.645497224367903) * x * sph_i[(31)]
    )
    sph_i[(44)] = -(0.9414688716912718) * (
        y * sph_i[(27)] - (1.63299316185545) * z * sph_i[(32)] + x * sph_i[(33)]
    )
    sph_i[(48)] = -(1.040832999733066) * (y * sph_i[(25)] - x * sph_i[(35)])


@njit(fastmath=True, cache=True)
def COMPUTE_SPH_DERIVATIVE_L6(
    x, y, z, x2, y2, z2, sph_i, dx_sph_i, dy_sph_i, dz_sph_i,
):
    dx_sph_i[(36)] = (6.244997998398398) * sph_i[(25)]
    dy_sph_i[(48)] = -dx_sph_i[(36)]
    dx_sph_i[(37)] = (5.700877125495690) * sph_i[(26)]
    dy_sph_i[(47)] = -dx_sph_i[(37)]
    dx_sph_i[(38)] = (
        -(8.07303841165959)
        * y
        * (
            y2 * y2
            - (4.188790204786391) * sph_i[(5)] * sph_i[(5)]
            - (2.642218198404007) * sph_i[(22)]
        )
    )
    dx_sph_i[(39)] = -(15.29705854077835) * (
        (y2 - z2) * sph_i[(10)] + (0.2611164839335468) * sph_i[(26)]
    )
    dx_sph_i[(40)] = (32.08092506951781) * (
        sph_i[(5)] * ((0.577350269189626) * y * sph_i[(5)] - z * sph_i[(6)])
        + (0.364182810197360) * y * y2 * sph_i[(6)]
        + (0.3169804496925759) * sph_i[(29)]
    )
    dx_sph_i[(41)] = -(2.430862174021989) * sph_i[(28)]
    dy_sph_i[(43)] = dx_sph_i[(41)]
    dx_sph_i[(42)] = -(4.210376791603422) * sph_i[(31)]
    dy_sph_i[(42)] = -(4.210376791603422) * sph_i[(29)]
    dx_sph_i[(43)] = (4.660970900149851) * (
        z2 * z * ((1.666666666666667) * y2 + z2 - (2.642218198404007) * sph_i[(6)])
        + (1.245553603643984) * y * sph_i[(19)]
        + (1.781383145961857) * sph_i[(30)]
    )
    dx_sph_i[(44)] = (14.73928415223878) * (
        x * (y2 - z2) * (2 * x2 - z2 - y2) + (0.2856568031469765) * sph_i[(35)]
    )
    dx_sph_i[(45)] = (3.122498999199199) * (
        y * sph_i[(17)]
        - (1.224744871391589) * z2 * sph_i[(14)]
        + +(1.846372364689991) * sph_i[(32)]
    )
    tmp = (1.612451549659710) * (
        y * sph_i[(16)] - (1.4142135623730950) * z * sph_i[(23)]
    )
    dx_sph_i[(46)] = tmp + (6.18796485857095) * sph_i[(33)]
    dy_sph_i[(38)] = -tmp + (4.125309905713972) * sph_i[(33)]
    dx_sph_i[(47)] = (5.700877125495690) * sph_i[(34)]
    dx_sph_i[(48)] = (6.244997998398398) * sph_i[(35)]
    dy_sph_i[(36)] = dx_sph_i[(48)]
    dy_sph_i[(37)] = dx_sph_i[(47)]
    dy_sph_i[(39)] = -(3.122498999199199) * (
        -(1.22474487139159) * z2 * sph_i[(14)]
        + y * sph_i[(17)]
        - (1.10782341881399) * sph_i[(32)]
    )
    dy_sph_i[(40)] = (11.68332144554792) * (
        x * (-(1.585330919042404) * sph_i[(5)] * sph_i[(5)] + (z2 - y2) * sph_i[(6)])
        + (0.1740776559556978) * sph_i[(31)]
    )
    dy_sph_i[(41)] = (
        -(6.99145635022478)
        * z
        * (
            z2 * z2
            + ((5.28443639680801) * y2 - (4.188790204786391) * sph_i[(6)]) * sph_i[(6)]
        )
    )
    dy_sph_i[(44)] = (13.49073756323204) * (
        y2 * z * sph_i[(5)]
        + (-(0.14940357616680) * x2 + (0.44821072850040) * y2 - (0.59761430466720) * z2)
        * sph_i[(11)]
    )
    dy_sph_i[(45)] = (
        (7.648529270389177) * (y2 - z2 - (1.58533091904240) * sph_i[(6)]) * sph_i[(10)]
    )
    dy_sph_i[(46)] = (11.40175425099138) * (
        (0.2360174359706574) * y2 * y2 * y
        + (y2 - 3 * z2) * sph_i[(9)]
        + (0.1348399724926484) * sph_i[(25)]
    )
    dz_sph_i[(36)] = 0.0
    dz_sph_i[(37)] = (3.605551275463989) * sph_i[(25)]
    dz_sph_i[(38)] = (4.861724348043977) * sph_i[(26)]
    dz_sph_i[(39)] = (5.64881323014763) * sph_i[(27)]
    dz_sph_i[(40)] = (6.14964891828646) * sph_i[(28)]
    dz_sph_i[(41)] = (6.43145678393600) * sph_i[(29)]
    dz_sph_i[(42)] = (6.52268767805531) * sph_i[(30)]
    dz_sph_i[(43)] = (6.43145678393600) * sph_i[(31)]
    dz_sph_i[(44)] = (6.14964891828646) * sph_i[(32)]
    dz_sph_i[(45)] = (5.64881323014763) * sph_i[(33)]
    dz_sph_i[(46)] = (4.861724348043977) * sph_i[(34)]
    dz_sph_i[(47)] = (3.605551275463989) * sph_i[(35)]
    dz_sph_i[(48)] = 0.0


"""
Combines the macro hard-coded Ylm calculators to get all the terms up to a
given value. Macro version. This uses if  to decide at compile time
which macro(s) should be called
"""

@njit(fastmath=True, cache=True)
def SPH0(x, y, z, x2, y2, z2, sph_i,):
    COMPUTE_SPH_L0( sph_i,)

@njit(fastmath=True, cache=True)
def SPH1(x, y, z, x2, y2, z2, sph_i,):
    COMPUTE_SPH_L0( sph_i,)
    COMPUTE_SPH_L1( x, y, z, sph_i,)

@njit(fastmath=True, cache=True)
def SPH2(x, y, z, x2, y2, z2, sph_i,):
    COMPUTE_SPH_L0( sph_i,)
    COMPUTE_SPH_L1( x, y, z, sph_i,)
    COMPUTE_SPH_L2( x, y, z, x2, y2, z2, sph_i,)

@njit(fastmath=True, cache=True)
def SPH3(x, y, z, x2, y2, z2, sph_i,):
    COMPUTE_SPH_L0( sph_i,)
    COMPUTE_SPH_L1( x, y, z, sph_i,)
    COMPUTE_SPH_L2( x, y, z, x2, y2, z2, sph_i,)
    COMPUTE_SPH_L3( x, y, z, x2, y2, z2, sph_i,)

@njit(fastmath=True, cache=True)
def SPH4(x, y, z, x2, y2, z2, sph_i,):
    COMPUTE_SPH_L0( sph_i,)
    COMPUTE_SPH_L1( x, y, z, sph_i,)
    COMPUTE_SPH_L2( x, y, z, x2, y2, z2, sph_i,)
    COMPUTE_SPH_L3( x, y, z, x2, y2, z2, sph_i,)
    COMPUTE_SPH_L4( x, y, z, x2, y2, z2, sph_i,)

@njit(fastmath=True, cache=True)
def SPH5(x, y, z, x2, y2, z2, sph_i,):
    COMPUTE_SPH_L0( sph_i,)
    COMPUTE_SPH_L1( x, y, z, sph_i,)
    COMPUTE_SPH_L2( x, y, z, x2, y2, z2, sph_i,)
    COMPUTE_SPH_L3( x, y, z, x2, y2, z2, sph_i,)
    COMPUTE_SPH_L4( x, y, z, x2, y2, z2, sph_i,)
    COMPUTE_SPH_L5( x, y, z, x2, y2, z2, sph_i,)

@njit(fastmath=True, cache=True)
def SPH6(x, y, z, x2, y2, z2, sph_i,):
    COMPUTE_SPH_L0( sph_i,)
    COMPUTE_SPH_L1( x, y, z, sph_i,)
    COMPUTE_SPH_L2( x, y, z, x2, y2, z2, sph_i,)
    COMPUTE_SPH_L3( x, y, z, x2, y2, z2, sph_i,)
    COMPUTE_SPH_L4( x, y, z, x2, y2, z2, sph_i,)
    COMPUTE_SPH_L5( x, y, z, x2, y2, z2, sph_i,)
    COMPUTE_SPH_L6( x, y, z, x2, y2, z2, sph_i,)

@njit(fastmath=True, cache=True)
def SPH0_GRAD(x, y, z, x2, y2, z2, sph_i, dsdx, dsdy, dsdz):
    SPH0( x, y, z, x2, y2, z2, sph_i,)
    COMPUTE_SPH_DERIVATIVE_L0( sph_i, dsdx, dsdy, dsdz)

@njit(fastmath=True, cache=True)
def SPH1_GRAD(x, y, z, x2, y2, z2, sph_i, dsdx, dsdy, dsdz):
    SPH1( x, y, z, x2, y2, z2, sph_i,)
    COMPUTE_SPH_DERIVATIVE_L0( sph_i, dsdx, dsdy, dsdz)
    COMPUTE_SPH_DERIVATIVE_L1( sph_i, dsdx, dsdy, dsdz)

@njit(fastmath=True, cache=True)
def SPH2_GRAD(x, y, z, x2, y2, z2, sph_i, dsdx, dsdy, dsdz):
    SPH2( x, y, z, x2, y2, z2, sph_i,)
    COMPUTE_SPH_DERIVATIVE_L0( sph_i, dsdx, dsdy, dsdz)
    COMPUTE_SPH_DERIVATIVE_L1( sph_i, dsdx, dsdy, dsdz)
    COMPUTE_SPH_DERIVATIVE_L2( x, y, z, x2, y2, z2, sph_i, dsdx, dsdy, dsdz)

@njit(fastmath=True, cache=True)
def SPH3_GRAD(x, y, z, x2, y2, z2, sph_i, dsdx, dsdy, dsdz):
    SPH3( x, y, z, x2, y2, z2, sph_i,)
    COMPUTE_SPH_DERIVATIVE_L0( sph_i, dsdx, dsdy, dsdz)
    COMPUTE_SPH_DERIVATIVE_L1( sph_i, dsdx, dsdy, dsdz)
    COMPUTE_SPH_DERIVATIVE_L2( x, y, z, x2, y2, z2, sph_i, dsdx, dsdy, dsdz)
    COMPUTE_SPH_DERIVATIVE_L3( x, y, z, x2, y2, z2, sph_i, dsdx, dsdy, dsdz)

@njit(fastmath=True, cache=True)
def SPH4_GRAD(x, y, z, x2, y2, z2, sph_i, dsdx, dsdy, dsdz):
    SPH4( x, y, z, x2, y2, z2, sph_i,)
    COMPUTE_SPH_DERIVATIVE_L0( sph_i, dsdx, dsdy, dsdz)
    COMPUTE_SPH_DERIVATIVE_L1( sph_i, dsdx, dsdy, dsdz)
    COMPUTE_SPH_DERIVATIVE_L2( x, y, z, x2, y2, z2, sph_i, dsdx, dsdy, dsdz)
    COMPUTE_SPH_DERIVATIVE_L3( x, y, z, x2, y2, z2, sph_i, dsdx, dsdy, dsdz)
    COMPUTE_SPH_DERIVATIVE_L4( x, y, z, x2, y2, z2, sph_i, dsdx, dsdy, dsdz)

@njit(fastmath=True, cache=True)
def SPH5_GRAD(x, y, z, x2, y2, z2, sph_i, dsdx, dsdy, dsdz):
    SPH5( x, y, z, x2, y2, z2, sph_i,)
    COMPUTE_SPH_DERIVATIVE_L0( sph_i, dsdx, dsdy, dsdz)
    COMPUTE_SPH_DERIVATIVE_L1( sph_i, dsdx, dsdy, dsdz)
    COMPUTE_SPH_DERIVATIVE_L2( x, y, z, x2, y2, z2, sph_i, dsdx, dsdy, dsdz)
    COMPUTE_SPH_DERIVATIVE_L3( x, y, z, x2, y2, z2, sph_i, dsdx, dsdy, dsdz)
    COMPUTE_SPH_DERIVATIVE_L4( x, y, z, x2, y2, z2, sph_i, dsdx, dsdy, dsdz)
    COMPUTE_SPH_DERIVATIVE_L5( x, y, z, x2, y2, z2, sph_i, dsdx, dsdy, dsdz)

@njit(fastmath=True, cache=True)
def SPH6_GRAD(x, y, z, x2, y2, z2, sph_i, dsdx, dsdy, dsdz):
    SPH6( x, y, z, x2, y2, z2, sph_i,)
    COMPUTE_SPH_DERIVATIVE_L0( sph_i, dsdx, dsdy, dsdz)
    COMPUTE_SPH_DERIVATIVE_L1( sph_i, dsdx, dsdy, dsdz)
    COMPUTE_SPH_DERIVATIVE_L2( x, y, z, x2, y2, z2, sph_i, dsdx, dsdy, dsdz)
    COMPUTE_SPH_DERIVATIVE_L3( x, y, z, x2, y2, z2, sph_i, dsdx, dsdy, dsdz)
    COMPUTE_SPH_DERIVATIVE_L4( x, y, z, x2, y2, z2, sph_i, dsdx, dsdy, dsdz)
    COMPUTE_SPH_DERIVATIVE_L5( x, y, z, x2, y2, z2, sph_i, dsdx, dsdy, dsdz)
    COMPUTE_SPH_DERIVATIVE_L6( x, y, z, x2, y2, z2, sph_i, dsdx, dsdy, dsdz)


@njit(fastmath=True, cache=True)
def HARDCODED_SPH_MACRO( HARDCODED_LMAX, x, y, z, x2, y2, z2, sph_i,):
    assert HARDCODED_LMAX <= SPHERICART_LMAX_HARDCODED, "Computing hardcoded sph beyond what is currently implemented."

    if HARDCODED_LMAX == 0: SPH0(x, y, z, x2, y2, z2, sph_i)
    elif HARDCODED_LMAX == 1: SPH1(x, y, z, x2, y2, z2, sph_i)
    elif HARDCODED_LMAX == 2: SPH2(x, y, z, x2, y2, z2, sph_i)
    elif HARDCODED_LMAX == 3: SPH3(x, y, z, x2, y2, z2, sph_i)
    elif HARDCODED_LMAX == 4: SPH4(x, y, z, x2, y2, z2, sph_i)
    elif HARDCODED_LMAX == 5: SPH5(x, y, z, x2, y2, z2, sph_i)
    else: SPH6(x, y, z, x2, y2, z2, sph_i)
    return
    COMPUTE_SPH_L0( sph_i,)
    #if (HARDCODED_LMAX > 0):
    COMPUTE_SPH_L1( x, y, z, sph_i,)
    #if (HARDCODED_LMAX > 1):
    COMPUTE_SPH_L2( x, y, z, x2, y2, z2, sph_i,)
    if (HARDCODED_LMAX ==3):
        COMPUTE_SPH_L3( x, y, z, x2, y2, z2, sph_i,)
    elif (HARDCODED_LMAX ==4):
        COMPUTE_SPH_L3( x, y, z, x2, y2, z2, sph_i,)
        COMPUTE_SPH_L4( x, y, z, x2, y2, z2, sph_i,)
    elif (HARDCODED_LMAX ==5):
        COMPUTE_SPH_L3( x, y, z, x2, y2, z2, sph_i,)
        COMPUTE_SPH_L4( x, y, z, x2, y2, z2, sph_i,)
        COMPUTE_SPH_L5( x, y, z, x2, y2, z2, sph_i,)
    elif (HARDCODED_LMAX ==6):
        COMPUTE_SPH_L3( x, y, z, x2, y2, z2, sph_i,)
        COMPUTE_SPH_L4( x, y, z, x2, y2, z2, sph_i,)
        COMPUTE_SPH_L5( x, y, z, x2, y2, z2, sph_i,)
        COMPUTE_SPH_L6( x, y, z, x2, y2, z2, sph_i,)


@njit(fastmath=True, cache=True)
def HARDCODED_SPH_DERIVATIVE_MACRO(
    HARDCODED_LMAX, x, y, z, x2, y2, z2, sph_i, dx_sph_i, dy_sph_i, dz_sph_i,
):
    if HARDCODED_LMAX == 0: SPH0_GRAD(x, y, z, x2, y2, z2, sph_i, dx_sph_i, dy_sph_i, dz_sph_i)
    elif HARDCODED_LMAX == 1: SPH1_GRAD(x, y, z, x2, y2, z2, sph_i, dx_sph_i, dy_sph_i, dz_sph_i)
    elif HARDCODED_LMAX == 2: SPH2_GRAD(x, y, z, x2, y2, z2, sph_i, dx_sph_i, dy_sph_i, dz_sph_i)
    elif HARDCODED_LMAX == 3: SPH3_GRAD(x, y, z, x2, y2, z2, sph_i, dx_sph_i, dy_sph_i, dz_sph_i)
    elif HARDCODED_LMAX == 4: SPH4_GRAD(x, y, z, x2, y2, z2, sph_i, dx_sph_i, dy_sph_i, dz_sph_i)
    elif HARDCODED_LMAX == 5: SPH5_GRAD(x, y, z, x2, y2, z2, sph_i, dx_sph_i, dy_sph_i, dz_sph_i)
    else: SPH6_GRAD(x, y, z, x2, y2, z2, sph_i, dx_sph_i, dy_sph_i, dz_sph_i)
    return
        

    COMPUTE_SPH_DERIVATIVE_L0(
        sph_i, dx_sph_i, dy_sph_i, dz_sph_i,
    )
    COMPUTE_SPH_DERIVATIVE_L1(
        sph_i, dx_sph_i, dy_sph_i, dz_sph_i,
    )
    COMPUTE_SPH_DERIVATIVE_L2(
        x, y, z, x2, y2, z2, sph_i, dx_sph_i, dy_sph_i, dz_sph_i,
    )
    if (HARDCODED_LMAX == 3):
        COMPUTE_SPH_DERIVATIVE_L3(
            x, y, z, x2, y2, z2, sph_i, dx_sph_i, dy_sph_i, dz_sph_i,
        )
    if (HARDCODED_LMAX == 4):
        COMPUTE_SPH_DERIVATIVE_L4(
            x, y, z, x2, y2, z2, sph_i, dx_sph_i, dy_sph_i, dz_sph_i,
        )
    if (HARDCODED_LMAX == 5):
        COMPUTE_SPH_DERIVATIVE_L5(
            x, y, z, x2, y2, z2, sph_i, dx_sph_i, dy_sph_i, dz_sph_i,
        )
    if (HARDCODED_LMAX == 6):
        COMPUTE_SPH_DERIVATIVE_L3(
            x, y, z, x2, y2, z2, sph_i, dx_sph_i, dy_sph_i, dz_sph_i,
        )
        COMPUTE_SPH_DERIVATIVE_L4(
            x, y, z, x2, y2, z2, sph_i, dx_sph_i, dy_sph_i, dz_sph_i,
        )
        COMPUTE_SPH_DERIVATIVE_L5(
            x, y, z, x2, y2, z2, sph_i, dx_sph_i, dy_sph_i, dz_sph_i,
        )
        COMPUTE_SPH_DERIVATIVE_L6(
            x, y, z, x2, y2, z2, sph_i, dx_sph_i, dy_sph_i, dz_sph_i,
        )

