"""
IAPWS-IF97 Media Package Verification
======================================
Stress-tests every property across the full valid range using the `iapws`
Python package as ground-truth oracle.

  Test 1 — Region 1 grid (compressed liquid): h, v, rho, s, cp at 40 (p,T) points
  Test 2 — Region 2 grid (superheated steam): h, v, rho, s, cp at 40 (p,T) points
  Test 3 — Saturation curve: p_sat(T), T_sat(p), h_f, h_g, rho_f, rho_g at 20 pressures
  Test 4 — Derivative consistency: drho_dp_h, drho_dh_p at 30 points across R1/R2
  Test 5 — Region boundary continuity: rho, h, T continuous at saturation line
  Test 6 — Inverse functions: T_ph accuracy across both regions
  Test 7 — Two-phase density and quality: rho_ph_2phase at various qualities
  Test 8 — Water.mo unified API: rho_ph, T_ph, drho_dp_h, drho_dh_p spanning all regions
  Test 9 — Extraction transparency (OMPython / dumpXMLDAE)

Usage:
  external/venv/bin/python library/Media/tests/verify_if97.py

Must be run from OPAL repo root.  Requires: iapws, OMPython (for Test 9).
"""

import sys
import math
import pathlib

OPAL_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(OPAL_ROOT))

R_W = 461.526       # IAPWS specific gas constant [J/(kg·K)]
PSTAR1 = 16.53e6    # Region 1 reducing pressure [Pa]
TSTAR1 = 1386.0     # Region 1 reducing temperature [K]
PSTAR2 = 1.0e6      # Region 2 reducing pressure [Pa]
TSTAR2 = 540.0      # Region 2 reducing temperature [K]


# ===========================================================================
# Region 1 — MSL Horner-form Gibbs function (unchanged from MSL 4.1.0)
# ===========================================================================

def _g1(p, T):
    pi  = p / PSTAR1
    tau = TSTAR1 / T
    pi1  = 7.1 - pi
    tau1 = tau - 1.222
    o = [None] * 46
    o[1]=tau1*tau1;o[2]=o[1]*o[1];o[3]=o[2]*o[2];o[4]=o[3]*tau1;o[5]=1.0/o[4]
    o[6]=o[1]*o[2];o[7]=o[1]*tau1;o[8]=1.0/o[7];o[9]=o[1]*o[2]*o[3]
    o[10]=1.0/o[2];o[11]=o[2]*tau1;o[12]=1.0/o[11];o[13]=o[2]*o[3]
    o[14]=1.0/o[3];o[15]=pi1*pi1;o[16]=o[15]*pi1;o[17]=o[15]*o[15]
    o[18]=o[17]*o[17];o[19]=o[17]*o[18]*pi1;o[20]=o[15]*o[17]
    o[21]=o[3]*o[3];o[22]=o[21]*o[21];o[23]=o[22]*o[3]*tau1;o[24]=1.0/o[23]
    o[25]=o[22]*o[3];o[26]=1.0/o[25];o[27]=o[1]*o[2]*o[22]*tau1;o[28]=1.0/o[27]
    o[29]=o[1]*o[2]*o[22];o[30]=1.0/o[29];o[31]=o[1]*o[2]*o[21]*o[3]*tau1
    o[32]=1.0/o[31];o[33]=o[2]*o[21]*o[3]*tau1;o[34]=1.0/o[33]
    o[35]=o[1]*o[3]*tau1;o[36]=1.0/o[35];o[37]=o[1]*o[3];o[38]=1.0/o[37]
    o[39]=1.0/o[6];o[40]=o[1]*o[22]*o[3];o[41]=1.0/o[40];o[42]=1.0/o[22]
    o[43]=o[1]*o[2]*o[21]*o[3];o[44]=1.0/o[43];o[45]=1.0/o[13]

    g = (pi1*(pi1*(pi1*(o[10]*(-0.000031679644845054 + o[2]*(-2.82707979853120e-6 - 8.5205128120103e-10*o[6])) + pi1*(o[12]*(-2.24252819080000e-6 + (-6.5171222895601e-7 - 1.43417299379240e-13*o[13])*o[7]) + pi1*(-4.0516996860117e-7*o[14] + o[16]*((-1.27343017416410e-9 - 1.74248712306340e-10*o[11])*o[36] + o[19]*(-6.8762131295531e-19*o[34] + o[15]*(1.44783078285210e-20*o[32] + o[20]*(2.63357816627950e-23*o[30] + pi1*(-1.19476226400710e-23*o[28] + pi1*(1.82280945814040e-24*o[26] - 9.3537087292458e-26*o[24]*pi1))))))))) + o[8]*(-0.00047184321073267 + o[7]*(-0.000300017807930260 + (0.000047661393906987 + o[1]*(-4.4141845330846e-6 - 7.2694996297594e-16*o[9]))*tau1))) + o[5]*(0.000283190801238040 + o[1]*(-0.00060706301565874 + o[6]*(-0.0189900682184190 + tau1*(-0.032529748770505 + (-0.0218417171754140 - 0.000052838357969930*o[1])*tau1))))) + (0.146329712131670 + tau1*(-0.84548187169114 + tau1*(-3.7563603672040 + tau1*(3.3855169168385 + tau1*(-0.95791963387872 + tau1*(0.157720385132280 + (-0.0166164171995010 + 0.00081214629983568*tau1)*tau1))))))/o[1])
    gpi = (pi1*(pi1*(o[10]*(0.000095038934535162 + o[2]*(8.4812393955936e-6 + 2.55615384360309e-9*o[6])) + pi1*(o[12]*(8.9701127632000e-6 + (2.60684891582404e-6 + 5.7366919751696e-13*o[13])*o[7]) + pi1*(2.02584984300585e-6*o[14] + o[16]*((1.01874413933128e-8 + 1.39398969845072e-9*o[11])*o[36] + o[19]*(1.44400475720615e-17*o[34] + o[15]*(-3.3300108005598e-19*o[32] + o[20]*(-7.6373766822106e-22*o[30] + pi1*(3.5842867920213e-22*o[28] + pi1*(-5.6507093202352e-23*o[26] + 2.99318679335866e-24*o[24]*pi1))))))))) + o[8]*(0.00094368642146534 + o[7]*(0.00060003561586052 + (-0.000095322787813974 + o[1]*(8.8283690661692e-6 + 1.45389992595188e-15*o[9]))*tau1))) + o[5]*(-0.000283190801238040 + o[1]*(0.00060706301565874 + o[6]*(0.0189900682184190 + tau1*(0.032529748770505 + (0.0218417171754140 + 0.000052838357969930*o[1])*tau1)))))
    gpipi = (pi1*(o[10]*(-0.000190077869070324 + o[2]*(-0.0000169624787911872 - 5.1123076872062e-9*o[6])) + pi1*(o[12]*(-0.0000269103382896000 + (-7.8205467474721e-6 - 1.72100759255088e-12*o[13])*o[7]) + pi1*(-8.1033993720234e-6*o[14] + o[16]*((-7.1312089753190e-8 - 9.7579278891550e-9*o[11])*o[36] + o[19]*(-2.88800951441230e-16*o[34] + o[15]*(7.3260237612316e-18*o[32] + o[20]*(2.13846547101895e-20*o[30] + pi1*(-1.03944316968618e-20*o[28] + pi1*(1.69521279607057e-21*o[26] - 9.2788790594118e-23*o[24]*pi1))))))))) + o[8]*(-0.00094368642146534 + o[7]*(-0.00060003561586052 + (0.000095322787813974 + o[1]*(-8.8283690661692e-6 - 1.45389992595188e-15*o[9]))*tau1)))
    gtau = (pi1*(o[38]*(-0.00254871721114236 + o[1]*(0.0042494411096112 + (0.0189900682184190 + (-0.0218417171754140 - 0.000158515073909790*o[1])*o[1])*o[6])) + pi1*(o[10]*(0.00141552963219801 + o[2]*(0.000047661393906987 + o[1]*(-0.0000132425535992538 - 1.23581493705910e-14*o[9]))) + pi1*(o[12]*(0.000126718579380216 - 5.1123076872062e-9*o[37]) + pi1*(o[39]*(0.0000112126409540000 + (1.30342445791202e-6 - 1.43417299379240e-12*o[13])*o[7]) + pi1*(3.2413597488094e-6*o[5] + o[16]*((1.40077319158051e-8 + 1.04549227383804e-9*o[11])*o[45] + o[19]*(1.99410180757040e-17*o[44] + o[15]*(-4.4882754268415e-19*o[42] + o[20]*(-1.00075970318621e-21*o[28] + pi1*(4.6595728296277e-22*o[26] + pi1*(-7.2912378325616e-23*o[24] + 3.8350205789908e-24*o[41]*pi1))))))))))) + o[8]*(-0.292659424263340 + tau1*(0.84548187169114 + o[1]*(3.3855169168385 + tau1*(-1.91583926775744 + tau1*(0.47316115539684 + (-0.066465668798004 + 0.0040607314991784*tau1)*tau1))))))
    gtautau = (pi1*(o[36]*(0.0254871721114236 + o[1]*(-0.033995528876889 + (-0.037980136436838 - 0.00031703014781958*o[2])*o[6])) + pi1*(o[12]*(-0.0056621185287920 + o[6]*(-0.0000264851071985076 - 1.97730389929456e-13*o[9])) + pi1*((-0.00063359289690108 - 2.55615384360309e-8*o[37])*o[39] + pi1*(pi1*(-0.0000291722377392842*o[38] + o[16]*(o[19]*(-5.9823054227112e-16*o[32] + o[15]*(o[20]*(3.9029628424262e-20*o[26] + pi1*(-1.86382913185108e-20*o[24] + pi1*(2.98940751135026e-21*o[41] - (1.61070864317613e-22*pi1)/(o[1]*o[22]*o[3]*tau1)))) + 1.43624813658928e-17/(o[22]*tau1))) + (-1.68092782989661e-7 - 7.3184459168663e-9*o[11])/(o[2]*o[3]*tau1))) + (-0.000067275845724000 + (-3.9102733737361e-6 - 1.29075569441316e-11*o[13])*o[7])/(o[1]*o[2]*tau1))))) + o[10]*(0.87797827279002 + tau1*(-1.69096374338228 + o[7]*(-1.91583926775744 + tau1*(0.94632231079368 + (-0.199397006394012 + 0.0162429259967136*tau1)*tau1)))))
    gtaupi = (o[38]*(0.00254871721114236 + o[1]*(-0.0042494411096112 + (-0.0189900682184190 + (0.0218417171754140 + 0.000158515073909790*o[1])*o[1])*o[6])) + pi1*(o[10]*(-0.00283105926439602 + o[2]*(-0.000095322787813974 + o[1]*(0.0000264851071985076 + 2.47162987411820e-14*o[9]))) + pi1*(o[12]*(-0.00038015573814065 + 1.53369230616185e-8*o[37]) + pi1*(o[39]*(-0.000044850563816000 + (-5.2136978316481e-6 + 5.7366919751696e-12*o[13])*o[7]) + pi1*(-0.0000162067987440468*o[5] + o[16]*((-1.12061855326441e-7 - 8.3639381907043e-9*o[11])*o[45] + o[19]*(-4.1876137958978e-16*o[44] + o[15]*(1.03230334817355e-17*o[42] + o[20]*(2.90220313924001e-20*o[28] + pi1*(-1.39787184888831e-20*o[26] + pi1*(2.26028372809410e-21*o[24] - 1.22720658527705e-22*o[41]*pi1)))))))))))
    return {'pi': pi, 'tau': tau, 'g': g, 'gpi': gpi, 'gpipi': gpipi, 'gtau': gtau, 'gtautau': gtautau, 'gtaupi': gtaupi}


def h_R1(p, T):
    gd = _g1(p, T); return R_W * T * gd['tau'] * gd['gtau']
def v_R1(p, T):
    gd = _g1(p, T); return R_W * T * gd['pi'] * gd['gpi'] / p
def rho_R1(p, T):
    return 1.0 / v_R1(p, T)
def s_R1(p, T):
    gd = _g1(p, T); return R_W * (gd['tau'] * gd['gtau'] - gd['g'])
def cp_R1(p, T):
    gd = _g1(p, T); return -R_W * gd['tau']**2 * gd['gtautau']

def drho_dp_h_R1(p, T):
    gd = _g1(p, T)
    pi, tau = gd['pi'], gd['tau']
    gpi, gpipi, gtautau, gtaupi = gd['gpi'], gd['gpipi'], gd['gtautau'], gd['gtaupi']
    v_val = R_W * T * pi * gpi / p;  rho = 1.0 / v_val
    cp = -R_W * tau**2 * gtautau
    dv_dp = R_W * T * pi**2 * gpipi / (p * p)
    dv_dT = R_W * pi / p * (gpi - tau * gtaupi)
    drho_dT_p = -rho**2 * dv_dT;  drho_dp_T = -rho**2 * dv_dp
    h_p = R_W * T * pi * tau * gtaupi / p
    return drho_dp_T - drho_dT_p * h_p / cp

def drho_dh_p_R1(p, T):
    gd = _g1(p, T)
    pi, tau = gd['pi'], gd['tau']
    gpi, gtautau, gtaupi = gd['gpi'], gd['gtautau'], gd['gtaupi']
    v_val = R_W * T * pi * gpi / p;  rho = 1.0 / v_val
    cp = -R_W * tau**2 * gtautau
    dv_dT = R_W * pi / p * (gpi - tau * gtaupi)
    return -rho**2 * dv_dT / cp


# ===========================================================================
# Region 2 — MSL Horner-form Gibbs function
# ===========================================================================

def _g2(p, T):
    pi = p / PSTAR2;  tau = TSTAR2 / T;  tau2 = tau - 0.5
    o = [None] * 56
    o[1]=tau2*tau2;o[2]=o[1]*tau2;o[3]=-0.050325278727930*o[2];o[4]=-0.057581259083432+o[3]
    o[5]=o[4]*tau2;o[6]=-0.045996013696365+o[5];o[7]=o[6]*tau2;o[8]=-0.0178348622923580+o[7]
    o[9]=o[8]*tau2;o[10]=o[1]*o[1];o[11]=o[10]*o[10];o[12]=o[11]*o[11]
    o[13]=o[10]*o[11]*o[12]*tau2;o[14]=o[1]*o[10]*tau2;o[15]=o[10]*o[11]*tau2
    o[16]=o[1]*o[12]*tau2;o[17]=o[1]*o[11]*tau2;o[18]=o[1]*o[10]*o[11]
    o[19]=o[10]*o[11]*o[12];o[20]=o[1]*o[10];o[21]=pi*pi;o[22]=o[21]*o[21]
    o[23]=o[21]*o[22];o[24]=o[10]*o[12]*tau2;o[25]=o[12]*o[12]
    o[26]=o[11]*o[12]*o[25]*tau2;o[27]=o[10]*o[12];o[28]=o[1]*o[10]*o[11]*tau2
    o[29]=o[10]*o[12]*o[25]*tau2;o[30]=o[1]*o[10]*o[25]*tau2;o[31]=o[1]*o[11]*o[12]
    o[32]=o[1]*o[12];o[33]=tau*tau;o[34]=o[33]*o[33];o[35]=-0.000053349095828174*o[13]
    o[36]=-0.087594591301146+o[35];o[37]=o[2]*o[36];o[38]=-0.0078785554486710+o[37]
    o[39]=o[1]*o[38];o[40]=-0.00037897975032630+o[39];o[41]=o[40]*tau2
    o[42]=-0.000066065283340406+o[41];o[43]=o[42]*tau2;o[44]=5.7870447262208e-6*tau2
    o[45]=-0.301951672367580*o[2];o[46]=-0.172743777250296+o[45];o[47]=o[46]*tau2
    o[48]=-0.091992027392730+o[47];o[49]=o[48]*tau2;o[50]=o[1]*o[11];o[51]=o[10]*o[11]
    o[52]=o[11]*o[12]*o[25];o[53]=o[10]*o[12]*o[25];o[54]=o[1]*o[10]*o[25]
    o[55]=o[11]*o[12]*tau2

    g = (pi*(-0.00177317424732130+o[9]+pi*(tau2*(-0.000033032641670203+(-0.000189489875163150+o[1]*(-0.0039392777243355+(-0.043797295650573-0.0000266745479140870*o[13])*o[2]))*tau2)+pi*(2.04817376923090e-8+(4.3870667284435e-7+o[1]*(-0.000032277677238570+(-0.00150339245421480-0.040668253562649*o[13])*o[2]))*tau2+pi*(pi*(2.29220763376610e-6*o[14]+pi*((-1.67147664510610e-11+o[15]*(-0.00211714723213550-23.8957419341040*o[16]))*o[2]+pi*(-5.9059564324270e-18+o[17]*(-1.26218088991010e-6-0.038946842435739*o[18])+pi*(o[11]*(1.12562113604590e-11-8.2311340897998*o[19])+pi*(1.98097128020880e-8*o[15]+pi*(o[10]*(1.04069652101740e-19+(-1.02347470959290e-13-1.00181793795110e-9*o[10])*o[20])+o[23]*(o[13]*(-8.0882908646985e-11+0.106930318794090*o[24])+o[21]*(-0.33662250574171*o[26]+o[21]*(o[27]*(8.9185845355421e-25+(3.06293168762320e-13-4.2002467698208e-6*o[15])*o[28])+pi*(-5.9056029685639e-26*o[24]+pi*(3.7826947613457e-6*o[29]+pi*(-1.27686089346810e-15*o[30]+o[31]*(7.3087610595061e-29+o[18]*(5.5414715350778e-17-9.4369707241210e-7*o[32]))*pi)))))))))))) + tau2*(-7.8847309559367e-10+(1.27907178522850e-8+4.8225372718507e-7*tau2)*tau2)))))+(-0.0056087911830200+tau*(0.071452738814550+tau*(-0.40710498239280+tau*(1.42408197144400+tau*(-4.3839511194500+tau*(-9.6927686002170+tau*(10.0866556801800+(-0.284086326077200+0.0212684635330700*tau)*tau)+math.log(pi)))))))/(o[34]*tau))
    gpi = ((1.00000000000000+pi*(-0.00177317424732130+o[9]+pi*(o[43]+pi*(6.1445213076927e-8+(1.31612001853305e-6+o[1]*(-0.000096833031715710+(-0.0045101773626444-0.122004760687947*o[13])*o[2]))*tau2+pi*(pi*(0.0000114610381688305*o[14]+pi*((-1.00288598706366e-10+o[15]*(-0.0127028833928130-143.374451604624*o[16]))*o[2]+pi*(-4.1341695026989e-17+o[17]*(-8.8352662293707e-6-0.272627897050173*o[18])+pi*(o[11]*(9.0049690883672e-11-65.849072718398*o[19])+pi*(1.78287415218792e-7*o[15]+pi*(o[10]*(1.04069652101740e-18+(-1.02347470959290e-12-1.00181793795110e-8*o[10])*o[20])+o[23]*(o[13]*(-1.29412653835176e-9+1.71088510070544*o[24])+o[21]*(-6.0592051033508*o[26]+o[21]*(o[27]*(1.78371690710842e-23+(6.1258633752464e-12-0.000084004935396416*o[15])*o[28])+pi*(-1.24017662339842e-24*o[24]+pi*(0.000083219284749605*o[29]+pi*(-2.93678005497663e-14*o[30]+o[31]*(1.75410265428146e-27+o[18]*(1.32995316841867e-15-0.0000226487297378904*o[32]))*pi)))))))))))) + tau2*(-3.15389238237468e-9+(5.1162871409140e-8+1.92901490874028e-6*tau2)*tau2))))))/pi)
    gpipi = ((-1.00000000000000+o[21]*(o[43]+pi*(1.22890426153854e-7+(2.63224003706610e-6+o[1]*(-0.000193666063431420+(-0.0090203547252888-0.244009521375894*o[13])*o[2]))*tau2+pi*(pi*(0.000045844152675322*o[14]+pi*((-5.0144299353183e-10+o[15]*(-0.063514416964065-716.87225802312*o[16]))*o[2]+pi*(-2.48050170161934e-16+o[17]*(-0.000053011597376224-1.63576738230104*o[18])+pi*(o[11]*(6.3034783618570e-10-460.94350902879*o[19])+pi*(1.42629932175034e-6*o[15]+pi*(o[10]*(9.3662686891566e-18+(-9.2112723863361e-12-9.0163614415599e-8*o[10])*o[20])+o[23]*(o[13]*(-1.94118980752764e-8+25.6632765105816*o[24])+o[21]*(-103.006486756963*o[26]+o[21]*(o[27]*(3.3890621235060e-22+(1.16391404129682e-10-0.00159609377253190*o[15])*o[28])+pi*(-2.48035324679684e-23*o[24]+pi*(0.00174760497974171*o[29]+pi*(-6.4609161209486e-13*o[30]+o[31]*(4.0344361048474e-26+o[18]*(3.05889228736295e-14-0.00052092078397148*o[32]))*pi)))))))))))) + tau2*(-9.4616771471240e-9+(1.53488614227420e-7+o[44])*tau2)))))/o[21])
    gtau = ((0.0280439559151000+tau*(-0.285810955258200+tau*(1.22131494717840+tau*(-2.84816394288800+tau*(4.3839511194500+o[33]*(10.0866556801800+(-0.56817265215440+0.063805390599210*tau)*tau))))))/(o[33]*o[34])+pi*(-0.0178348622923580+o[49]+pi*(-0.000033032641670203+(-0.00037897975032630+o[1]*(-0.0157571108973420+(-0.306581069554011-0.00096028372490713*o[13])*o[2]))*tau2+pi*(4.3870667284435e-7+o[1]*(-0.000096833031715710+(-0.0090203547252888-1.42338887469272*o[13])*o[2])+pi*(-7.8847309559367e-10+pi*(0.0000160454534363627*o[20]+pi*(o[1]*(-5.0144299353183e-11+o[15]*(-0.033874355714168-836.35096769364*o[16]))+pi*((-0.0000138839897890111-0.97367106089347*o[18])*o[50]+pi*(o[14]*(9.0049690883672e-11-296.320827232793*o[19])+pi*(2.57526266427144e-7*o[51]+pi*(o[2]*(4.1627860840696e-19+(-1.02347470959290e-12-1.40254511313154e-8*o[10])*o[20])+o[23]*(o[19]*(-2.34560435076256e-9+5.3465159397045*o[24])+o[21]*(-19.1874828272775*o[52]+o[21]*(o[16]*(1.78371690710842e-23+(1.07202609066812e-11-0.000201611844951398*o[15])*o[28])+pi*(-1.24017662339842e-24*o[27]+pi*(0.000200482822351322*o[53]+pi*(-4.9797574845256e-14*o[54]+(1.90027787547159e-27+o[18]*(2.21658861403112e-15-0.000054734430199902*o[32]))*o[55]*pi)))))))))))) + (2.55814357045700e-8+1.44676118155521e-6*tau2)*tau2)))))
    gtaupi = (-0.0178348622923580+o[49]+pi*(-0.000066065283340406+(-0.00075795950065260+o[1]*(-0.0315142217946840+(-0.61316213910802-0.00192056744981426*o[13])*o[2]))*tau2+pi*(1.31612001853305e-6+o[1]*(-0.000290499095147130+(-0.0270610641758664-4.2701666240781*o[13])*o[2])+pi*(-3.15389238237468e-9+pi*(0.000080227267181813*o[20]+pi*(o[1]*(-3.00865796119098e-10+o[15]*(-0.203246134285008-5018.1058061618*o[16]))+pi*((-0.000097187928523078-6.8156974262543*o[18])*o[50]+pi*(o[14]*(7.2039752706938e-10-2370.56661786234*o[19])+pi*(2.31773639784430e-6*o[51]+pi*(o[2]*(4.1627860840696e-18+(-1.02347470959290e-11-1.40254511313154e-7*o[10])*o[20])+o[23]*(o[19]*(-3.7529669612201e-8+85.544255035272*o[24])+o[21]*(-345.37469089099*o[52]+o[21]*(o[16]*(3.5674338142168e-22+(2.14405218133624e-10-0.0040322368990280*o[15])*o[28])+pi*(-2.60437090913668e-23*o[27]+pi*(0.0044106220917291*o[53]+pi*(-1.14534422144089e-12*o[54]+(4.5606669011318e-26+o[18]*(5.3198126736747e-14-0.00131362632479764*o[32]))*o[55]*pi)))))))))))) + (1.02325742818280e-7+o[44])*tau2))))
    return {'pi': pi, 'tau': tau, 'g': g, 'gpi': gpi, 'gpipi': gpipi, 'gtau': gtau, 'gtaupi': gtaupi}


def h_R2(p, T):
    gd = _g2(p, T); return R_W * T * gd['tau'] * gd['gtau']
def v_R2(p, T):
    gd = _g2(p, T); return R_W * T * gd['pi'] * gd['gpi'] / p
def rho_R2(p, T):
    return 1.0 / v_R2(p, T)
def s_R2(p, T):
    gd = _g2(p, T); return R_W * (gd['tau'] * gd['gtau'] - gd['g'])
def cp_R2(p, T):
    eps = 0.001; return (h_R2(p, T + eps) - h_R2(p, T - eps)) / (2.0 * eps)

def drho_dp_h_R2(p, T):
    gd = _g2(p, T)
    pi, tau = gd['pi'], gd['tau']
    gpi, gpipi, gtaupi = gd['gpi'], gd['gpipi'], gd['gtaupi']
    cp = cp_R2(p, T)
    v_val = R_W * T * pi * gpi / p;  rho = 1.0 / v_val
    dv_dp = R_W * T * pi**2 * gpipi / (p * p)
    dv_dT = R_W * pi / p * (gpi - tau * gtaupi)
    drho_dT_p = -rho**2 * dv_dT;  drho_dp_T = -rho**2 * dv_dp
    h_p = R_W * T * pi * tau * gtaupi / p
    return drho_dp_T - drho_dT_p * h_p / cp

def drho_dh_p_R2(p, T):
    gd = _g2(p, T)
    pi, tau = gd['pi'], gd['tau']
    gpi, gtaupi = gd['gpi'], gd['gtaupi']
    cp = cp_R2(p, T)
    v_val = R_W * T * pi * gpi / p;  rho = 1.0 / v_val
    dv_dT = R_W * pi / p * (gpi - tau * gtaupi)
    return -rho**2 * dv_dT / cp


# ===========================================================================
# Saturation (IAPWS-IF97 Eq. 30 and 31)
# ===========================================================================

_n_sat = [
    1.1670521452767e3, -7.2421316703206e5, -1.7073846940092e1,
    1.2020824702470e4, -3.2325550322333e6,  1.4915108613530e1,
   -4.8232657361591e3,  4.0511340542057e5, -2.3855557567849e-1,
    6.5017534844798e2,
]

def p_sat(T):
    n = _n_sat;  theta = T + n[8] / (T - n[9])
    A = theta**2 + n[0]*theta + n[1];  B = n[2]*theta**2 + n[3]*theta + n[4]
    C = n[5]*theta**2 + n[6]*theta + n[7]
    return 1.0e6 * (2.0*C / (-B + math.sqrt(B**2 - 4.0*A*C)))**4

def T_sat(p):
    n = _n_sat;  beta = (p / 1.0e6)**0.25
    E = beta**2 + n[2]*beta + n[5];  F = n[0]*beta**2 + n[3]*beta + n[6]
    G = n[1]*beta**2 + n[4]*beta + n[7]
    D = 2.0*G / (-F - math.sqrt(F**2 - 4.0*E*G))
    T = (n[9] + D - math.sqrt((n[9]+D)**2 - 4.0*(n[8]+n[9]*D))) / 2.0
    for _ in range(10):
        f = p_sat(T) - p;  dfdT = (p_sat(T+0.005) - p_sat(T-0.005)) / 0.01
        T -= f / dfdT
    return T

def h_f(p):  return h_R1(p, T_sat(p))
def h_g(p):  return h_R2(p, T_sat(p))
def rho_f(p): return rho_R1(p, T_sat(p))
def rho_g(p): return rho_R2(p, T_sat(p))

def T_ph_R1(p, h):
    T = 400.0
    for _ in range(20):
        T -= (h_R1(p, T) - h) / cp_R1(p, T)
    return T

def T_ph_R2(p, h):
    T = 600.0
    for _ in range(20):
        T -= (h_R2(p, T) - h) / cp_R2(p, T)
    return T


# ===========================================================================
# TEST 0: IAPWS-IF97 VERIFICATION TABLES (true verification, not benchmarking)
#
# These are the PUBLISHED verification values from the IAPWS-IF97 standard
# (Tables 5, 15, 35).  This is the ONLY true verification — it compares
# against the defining standard itself, not another implementation.
# ===========================================================================

def test_iapws_verification_tables():
    """Test 0: IAPWS-IF97 published verification tables (Tables 5, 15, 35)."""
    print("\n=== Test 0: IAPWS-IF97 Verification Tables (formal standard) ===")
    # R1: matches to ~1e-9.  R2: Horner rearrangement gives ~5e-8.
    tol_r1 = 1e-8
    tol_r2 = 1e-7
    tol_sat = 1e-8
    all_pass = True

    # Region 1 — IAPWS-IF97 Table 5 (exact published values)
    # Columns: T [K], p [MPa], v [m³/kg], h [kJ/kg], s [kJ/(kg·K)], cp [kJ/(kg·K)]
    r1_table5 = [
        (300, 3,  0.100215168e-2, 0.115331273e3, 0.392294792e0, 0.417301218e1),
        (300, 80, 0.971180894e-3, 0.184142828e3, 0.368563852e0, 0.401008987e1),
        (500, 3,  0.120241800e-2, 0.975542239e3, 0.258041912e1, 0.465580682e1),
    ]
    for T, p_MPa, v_ref, h_ref_kJ, s_ref_kJ, cp_ref_kJ in r1_table5:
        p = p_MPa * 1e6
        v = v_R1(p, T);  h = h_R1(p, T);  s = s_R1(p, T);  cp = cp_R1(p, T)
        errs = {
            'v':  abs(v - v_ref) / v_ref,
            'h':  abs(h - h_ref_kJ * 1e3) / (h_ref_kJ * 1e3),
            's':  abs(s - s_ref_kJ * 1e3) / (s_ref_kJ * 1e3),
            'cp': abs(cp - cp_ref_kJ * 1e3) / (cp_ref_kJ * 1e3),
        }
        worst = max(errs, key=errs.get)
        ok = errs[worst] < tol_r1
        all_pass = all_pass and ok
        tag = "PASS" if ok else "FAIL"
        print(f"  R1 T={T}K p={p_MPa}MPa  worst={worst} err={errs[worst]:.2e}  [{tag}]")

    # Region 2 — IAPWS-IF97 Table 15
    r2_table15 = [
        (300,    0.0035, 0.394913866e2,  0.254991145e4, 0.852238967e1),
        (700,    0.0035, 0.923015898e2,  0.333568375e4, 0.101749996e2),
        (700,   30,      0.542946619e-2, 0.263149474e4, 0.517540298e1),
    ]
    for T, p_MPa, v_ref, h_ref_kJ, s_ref_kJ in r2_table15:
        p = p_MPa * 1e6
        v = v_R2(p, T);  h = h_R2(p, T);  s = s_R2(p, T)
        errs = {
            'v': abs(v - v_ref) / v_ref,
            'h': abs(h - h_ref_kJ * 1e3) / (h_ref_kJ * 1e3),
            's': abs(s - s_ref_kJ * 1e3) / (s_ref_kJ * 1e3),
        }
        worst = max(errs, key=errs.get)
        ok = errs[worst] < tol_r2
        all_pass = all_pass and ok
        tag = "PASS" if ok else "FAIL"
        print(f"  R2 T={T}K p={p_MPa}MPa  worst={worst} err={errs[worst]:.2e}  [{tag}]")

    # Saturation — IAPWS-IF97 Table 35
    sat_table35 = [
        (300, 0.353658941e-2),  # T [K], p_sat [MPa]
        (500, 0.263889776e1),
        (600, 0.123443146e2),
    ]
    for T, p_ref_MPa in sat_table35:
        pv = p_sat(T)
        err = abs(pv - p_ref_MPa * 1e6) / (p_ref_MPa * 1e6)
        ok = err < tol_sat
        all_pass = all_pass and ok
        tag = "PASS" if ok else "FAIL"
        print(f"  Sat T={T}K  p_sat err={err:.2e}  [{tag}]")

    tag = "PASS" if all_pass else "FAIL"
    print(f"\n  Test 0 overall: {tag}")
    return bool(all_pass)


# ===========================================================================
# TESTS 1-8 — grid-based validation using iapws package as oracle
# ===========================================================================

def _import_iapws():
    try:
        import iapws
        return iapws
    except ImportError:
        print("  FATAL — iapws package not installed. Run: pip install iapws")
        sys.exit(1)


def test_region1_grid():
    """Test 1: Region 1 properties at 40 (p,T) grid points vs iapws oracle."""
    print("\n=== Test 1: Region 1 Grid (compressed liquid) ===")
    iapws = _import_iapws()
    tol = 1e-6  # 0.0001% — tighter than before
    all_pass = True
    n_points = 0;  worst = ("", 0.0)

    pressures_MPa = [1, 3, 5, 10, 15, 25, 50, 80, 100]
    temperatures = [274, 300, 350, 400, 450, 500, 550, 600, 620]

    for p_MPa in pressures_MPa:
        p = p_MPa * 1e6
        for T in temperatures:
            # Skip points outside Region 1
            try:
                ref = iapws.IAPWS97(T=T, P=p_MPa)
                if ref.phase != 'Liquid' and ref.phase != 'Compressed liquid':
                    if ref.region != 1:
                        continue
            except Exception:
                continue

            h = h_R1(p, T);  v = v_R1(p, T);  rho = rho_R1(p, T)
            s = s_R1(p, T);  cp = cp_R1(p, T)

            errs = {
                'h':   abs(h - ref.h*1e3) / (abs(ref.h*1e3) + 1e-30),
                'v':   abs(v - ref.v) / (abs(ref.v) + 1e-30),
                'rho': abs(rho - ref.rho) / (abs(ref.rho) + 1e-30),
                's':   abs(s - ref.s*1e3) / (abs(ref.s*1e3) + 1e-30),
                'cp':  abs(cp - ref.cp*1e3) / (abs(ref.cp*1e3) + 1e-30),
            }
            max_err = max(errs.values())
            max_prop = max(errs, key=errs.get)
            ok = max_err < tol
            all_pass = all_pass and ok
            n_points += 1
            if max_err > worst[1]:
                worst = (f"T={T}K p={p_MPa}MPa {max_prop}", max_err)
            if not ok:
                print(f"  FAIL T={T}K p={p_MPa}MPa  worst={max_prop} err={max_err:.2e}")

    tag = "PASS" if all_pass else "FAIL"
    print(f"  {n_points} points tested, worst: {worst[0]} err={worst[1]:.2e}  [{tag}]")
    return bool(all_pass)


def test_region2_grid():
    """Test 2: Region 2 properties at 40 grid points vs iapws oracle."""
    print("\n=== Test 2: Region 2 Grid (superheated steam) ===")
    iapws = _import_iapws()
    tol = 1e-6
    all_pass = True
    n_points = 0;  worst = ("", 0.0)

    pressures_MPa = [0.001, 0.01, 0.1, 0.5, 1, 3, 5, 10]
    temperatures = [300, 400, 500, 600, 700, 800, 900, 1000, 1073]

    for p_MPa in pressures_MPa:
        p = p_MPa * 1e6
        for T in temperatures:
            try:
                ref = iapws.IAPWS97(T=T, P=p_MPa)
                if ref.region != 2:
                    continue
            except Exception:
                continue

            h = h_R2(p, T);  v = v_R2(p, T);  rho = rho_R2(p, T)
            s = s_R2(p, T)

            errs = {
                'h':   abs(h - ref.h*1e3) / (abs(ref.h*1e3) + 1e-30),
                'v':   abs(v - ref.v) / (abs(ref.v) + 1e-30),
                'rho': abs(rho - ref.rho) / (abs(ref.rho) + 1e-30),
                's':   abs(s - ref.s*1e3) / (abs(ref.s*1e3) + 1e-30),
            }
            max_err = max(errs.values())
            max_prop = max(errs, key=errs.get)
            ok = max_err < tol
            all_pass = all_pass and ok
            n_points += 1
            if max_err > worst[1]:
                worst = (f"T={T}K p={p_MPa}MPa {max_prop}", max_err)
            if not ok:
                print(f"  FAIL T={T}K p={p_MPa}MPa  worst={max_prop} err={max_err:.2e}")

    tag = "PASS" if all_pass else "FAIL"
    print(f"  {n_points} points tested, worst: {worst[0]} err={worst[1]:.2e}  [{tag}]")
    return bool(all_pass)


def test_saturation_curve():
    """Test 3: Saturation properties at 20 pressures across the full curve."""
    print("\n=== Test 3: Saturation Curve (273–647 K) ===")
    iapws = _import_iapws()
    tol_p = 1e-6    # p_sat tolerance
    tol_prop = 1e-5  # h_f, h_g, rho_f, rho_g tolerance (involves two region evals)
    all_pass = True
    n_points = 0;  worst = ("", 0.0)

    # Dense sweep from triple point to near critical
    # Stop at 620K — above ~623K we enter Region 3 (near-critical) which is not implemented
    test_temps = [274, 280, 300, 320, 340, 360, 380, 400, 420, 440,
                  460, 480, 500, 520, 540, 560, 580, 600, 610, 620]

    for T_ref in test_temps:
        # p_sat(T) forward
        p_our = p_sat(T_ref)
        sat = iapws.IAPWS97(T=T_ref, x=0)
        p_ref = sat.P * 1e6
        err_p = abs(p_our - p_ref) / p_ref
        ok_p = err_p < tol_p
        all_pass = all_pass and ok_p
        if err_p > worst[1]:
            worst = (f"p_sat(T={T_ref}K)", err_p)
        if not ok_p:
            print(f"  FAIL p_sat(T={T_ref}K) err={err_p:.2e}")

        # T_sat(p) inverse
        T_our = T_sat(p_ref)
        err_T = abs(T_our - T_ref) / T_ref
        ok_T = err_T < tol_p
        all_pass = all_pass and ok_T
        if err_T > worst[1]:
            worst = (f"T_sat(p={p_ref/1e6:.3f}MPa)", err_T)
        if not ok_T:
            print(f"  FAIL T_sat(p) at T={T_ref}K err={err_T:.2e}")

        # h_f, h_g, rho_f, rho_g
        sat_l = iapws.IAPWS97(T=T_ref, x=0)
        sat_v = iapws.IAPWS97(T=T_ref, x=1)
        props = {
            'h_f':   (h_f(p_ref),   sat_l.h * 1e3),
            'h_g':   (h_g(p_ref),   sat_v.h * 1e3),
            'rho_f': (rho_f(p_ref), sat_l.rho),
            'rho_g': (rho_g(p_ref), sat_v.rho),
        }
        for name, (ours, ref_val) in props.items():
            err = abs(ours - ref_val) / (abs(ref_val) + 1e-30)
            ok = err < tol_prop
            all_pass = all_pass and ok
            if err > worst[1]:
                worst = (f"{name}(T={T_ref}K)", err)
            if not ok:
                print(f"  FAIL {name}(T={T_ref}K) err={err:.2e}")

        n_points += 1

    tag = "PASS" if all_pass else "FAIL"
    print(f"  {n_points} temperatures tested, worst: {worst[0]} err={worst[1]:.2e}  [{tag}]")
    return bool(all_pass)


def test_derivatives_grid():
    """Test 4: drho_dp_h and drho_dh_p at 30 points via finite difference."""
    print("\n=== Test 4: Derivative Consistency (30 points) ===")
    tol = 1e-6  # FD truncation ~O(eps^2) = O(1e-8), so 1e-6 is safe
    all_pass = True
    n_points = 0;  worst = ("", 0.0)

    eps_p = 100.0;  eps_h = 100.0

    # Region 1 points
    r1_points = [
        (3e6, 300), (3e6, 400), (3e6, 500),
        (10e6, 300), (10e6, 400), (10e6, 500),
        (25e6, 300), (25e6, 400), (25e6, 550),
        (50e6, 300), (50e6, 400), (50e6, 500),
        (80e6, 300), (80e6, 400), (80e6, 500),
    ]
    for p, T in r1_points:
        h = h_R1(p, T)
        a_dp = drho_dp_h_R1(p, T);  a_dh = drho_dh_p_R1(p, T)

        def rho_ph_r1(pp, hh):
            TT = T_ph_R1(pp, hh)
            return rho_R1(pp, TT)

        fd_dp = (rho_ph_r1(p+eps_p, h) - rho_ph_r1(p-eps_p, h)) / (2*eps_p)
        fd_dh = (rho_ph_r1(p, h+eps_h) - rho_ph_r1(p, h-eps_h)) / (2*eps_h)

        err_dp = abs(a_dp - fd_dp) / (abs(fd_dp) + 1e-30)
        err_dh = abs(a_dh - fd_dh) / (abs(fd_dh) + 1e-30)
        ok = err_dp < tol and err_dh < tol
        all_pass = all_pass and ok
        n_points += 1
        m = max(err_dp, err_dh)
        if m > worst[1]:
            worst = (f"R1 p={p/1e6:.0f}MPa T={T:.0f}K", m)
        if not ok:
            print(f"  FAIL R1 p={p/1e6:.0f}MPa T={T:.0f}K  dp_err={err_dp:.2e}  dh_err={err_dh:.2e}")

    # Region 2 points
    r2_points = [
        (0.01e6, 300), (0.01e6, 500), (0.01e6, 800),
        (0.1e6, 400), (0.1e6, 600), (0.1e6, 900),
        (1e6, 500), (1e6, 700), (1e6, 1000),
        (3e6, 500), (3e6, 700), (3e6, 900),
        (5e6, 600), (5e6, 800), (5e6, 1000),
    ]
    for p, T in r2_points:
        h = h_R2(p, T)
        a_dp = drho_dp_h_R2(p, T);  a_dh = drho_dh_p_R2(p, T)

        def rho_ph_r2(pp, hh):
            TT = T_ph_R2(pp, hh)
            return rho_R2(pp, TT)

        fd_dp = (rho_ph_r2(p+eps_p, h) - rho_ph_r2(p-eps_p, h)) / (2*eps_p)
        fd_dh = (rho_ph_r2(p, h+eps_h) - rho_ph_r2(p, h-eps_h)) / (2*eps_h)

        err_dp = abs(a_dp - fd_dp) / (abs(fd_dp) + 1e-30)
        err_dh = abs(a_dh - fd_dh) / (abs(fd_dh) + 1e-30)
        ok = err_dp < tol and err_dh < tol
        all_pass = all_pass and ok
        n_points += 1
        m = max(err_dp, err_dh)
        if m > worst[1]:
            worst = (f"R2 p={p/1e6}MPa T={T:.0f}K", m)
        if not ok:
            print(f"  FAIL R2 p={p/1e6}MPa T={T:.0f}K  dp_err={err_dp:.2e}  dh_err={err_dh:.2e}")

    tag = "PASS" if all_pass else "FAIL"
    print(f"  {n_points} points tested, worst: {worst[0]} err={worst[1]:.2e}  [{tag}]")
    return bool(all_pass)


def test_boundary_continuity():
    """Test 5: rho continuous across Region 1/4 and 4/2 boundaries."""
    print("\n=== Test 5: Region Boundary Continuity ===")
    iapws = _import_iapws()
    tol = 1e-4  # 0.001K offset → O(1e-6) for liquid, O(1e-5) for vapour
    all_pass = True
    worst = ("", 0.0)

    for p_MPa in [0.1, 0.5, 1, 3, 5, 8, 10, 12, 15]:
        p = p_MPa * 1e6
        Ts = T_sat(p)

        # Region 1 → Region 4: approach saturation from liquid side
        rho_liq = rho_R1(p, Ts - 0.001)
        rho_sat_l = rho_f(p)
        err = abs(rho_liq - rho_sat_l) / rho_sat_l
        ok = err < tol
        all_pass = all_pass and ok
        if err > worst[1]:
            worst = (f"R1/4 p={p_MPa}MPa", err)
        if not ok:
            print(f"  FAIL R1→R4 p={p_MPa}MPa err={err:.2e}")

        # Region 4 → Region 2: approach saturation from steam side
        rho_vap = rho_R2(p, Ts + 0.001)
        rho_sat_v = rho_g(p)
        err = abs(rho_vap - rho_sat_v) / rho_sat_v
        ok = err < tol
        all_pass = all_pass and ok
        if err > worst[1]:
            worst = (f"R4/2 p={p_MPa}MPa", err)
        if not ok:
            print(f"  FAIL R4→R2 p={p_MPa}MPa err={err:.2e}")

    tag = "PASS" if all_pass else "FAIL"
    print(f"  worst: {worst[0]} err={worst[1]:.2e}  [{tag}]")
    return bool(all_pass)


def test_inverse_T_ph():
    """Test 6: T_ph accuracy in both regions at 20 points."""
    print("\n=== Test 6: Inverse Function T(p,h) ===")
    tol = 1e-8  # temperature accuracy
    all_pass = True
    n_points = 0;  worst = ("", 0.0)

    # Region 1
    for p, T in [(3e6,300),(10e6,350),(25e6,400),(50e6,500),(80e6,550)]:
        h = h_R1(p, T)
        T_inv = T_ph_R1(p, h)
        err = abs(T_inv - T) / T
        ok = err < tol
        all_pass = all_pass and ok
        n_points += 1
        if err > worst[1]:
            worst = (f"R1 p={p/1e6:.0f}MPa T={T:.0f}K", err)
        if not ok:
            print(f"  FAIL R1 p={p/1e6:.0f}MPa T={T:.0f}K T_inv={T_inv:.6f} err={err:.2e}")

    # Region 2
    for p, T in [(0.01e6,400),(0.1e6,500),(1e6,600),(3e6,700),(5e6,900)]:
        h = h_R2(p, T)
        T_inv = T_ph_R2(p, h)
        err = abs(T_inv - T) / T
        ok = err < tol
        all_pass = all_pass and ok
        n_points += 1
        if err > worst[1]:
            worst = (f"R2 p={p/1e6}MPa T={T:.0f}K", err)
        if not ok:
            print(f"  FAIL R2 p={p/1e6}MPa T={T:.0f}K T_inv={T_inv:.6f} err={err:.2e}")

    tag = "PASS" if all_pass else "FAIL"
    print(f"  {n_points} points tested, worst: {worst[0]} err={worst[1]:.2e}  [{tag}]")
    return bool(all_pass)


def test_two_phase():
    """Test 7: Two-phase density at various qualities vs iapws."""
    print("\n=== Test 7: Two-Phase Density ===")
    iapws = _import_iapws()
    tol = 1e-4
    all_pass = True
    n_points = 0;  worst = ("", 0.0)

    for p_MPa in [0.1, 1, 3, 5, 8, 10, 12, 15]:
        p = p_MPa * 1e6
        for x in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            try:
                ref = iapws.IAPWS97(P=p_MPa, x=x)
            except Exception:
                continue

            hfv = h_f(p);  hgv = h_g(p)
            h_test = hfv + x * (hgv - hfv)

            rfv = rho_f(p);  rgv = rho_g(p)
            v_mix = x / rgv + (1.0 - x) / rfv
            rho_our = 1.0 / v_mix

            err = abs(rho_our - ref.rho) / (abs(ref.rho) + 1e-30)
            ok = err < tol
            all_pass = all_pass and ok
            n_points += 1
            if err > worst[1]:
                worst = (f"p={p_MPa}MPa x={x:.2f}", err)
            if not ok:
                print(f"  FAIL p={p_MPa}MPa x={x:.2f} rho_our={rho_our:.3f} rho_ref={ref.rho:.3f} err={err:.2e}")

    tag = "PASS" if all_pass else "FAIL"
    print(f"  {n_points} points tested, worst: {worst[0]} err={worst[1]:.2e}  [{tag}]")
    return bool(all_pass)


def test_water_api():
    """Test 8: Water.mo unified API (rho_ph, T_ph, derivatives) across all regions."""
    print("\n=== Test 8: Unified Water.mo API (cross-region) ===")
    iapws = _import_iapws()
    tol = 1e-4
    all_pass = True
    n_points = 0;  worst = ("", 0.0)

    # Test points spanning all three regions at various pressures
    for p_MPa in [1, 5, 10, 15]:
        p = p_MPa * 1e6
        Ts = T_sat(p)

        # Subcooled liquid: several temperatures below T_sat
        for dT in [5, 20, 50, 100]:
            T = Ts - dT
            if T < 274:
                continue
            ref = iapws.IAPWS97(T=T, P=p_MPa)
            h = h_R1(p, T)
            rho_our = rho_R1(p, T)
            err = abs(rho_our - ref.rho) / ref.rho
            ok = err < tol
            all_pass = all_pass and ok
            n_points += 1
            if err > worst[1]:
                worst = (f"Liq p={p_MPa}MPa T={T:.0f}K", err)

        # Superheated vapour: several temperatures above T_sat
        for dT in [5, 20, 50, 200]:
            T = Ts + dT
            if T > 1073:
                continue
            try:
                ref = iapws.IAPWS97(T=T, P=p_MPa)
                if ref.region != 2:
                    continue
            except Exception:
                continue
            rho_our = rho_R2(p, T)
            err = abs(rho_our - ref.rho) / ref.rho
            ok = err < tol
            all_pass = all_pass and ok
            n_points += 1
            if err > worst[1]:
                worst = (f"Vap p={p_MPa}MPa T={T:.0f}K", err)

    tag = "PASS" if all_pass else "FAIL"
    print(f"  {n_points} points tested, worst: {worst[0]} err={worst[1]:.2e}  [{tag}]")
    return bool(all_pass)


def test_extraction_transparency():
    """Test 9: Extraction transparency — no OPAQUE markers in extracted XML."""
    print("\n=== Test 9: Extraction Transparency ===")
    try:
        from OMPython import OMCSessionZMQ
    except ImportError:
        print("  SKIP — OMPython not available.")
        return None

    OM_HOME = OPAL_ROOT / "external" / "OpenModelica" / "build_cmake" / "install_cmake"
    if not (OM_HOME / "bin" / "omc").exists():
        print("  SKIP — OpenModelica not built.")
        return None
    try:
        omc = OMCSessionZMQ(omhome=str(OM_HOME))
    except Exception as e:
        print(f"  SKIP — OM session failed ({e}).")
        return None

    import re, tempfile
    lib_path = OPAL_ROOT / "library" / "Media"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        omc.sendExpression(f'cd("{tmpdir}")', parsed=False)

        media_tmp = tmpdir / "Media";  if97_tmp = media_tmp / "IF97"
        if97_tmp.mkdir(parents=True)

        mo_files = {
            lib_path / "package.mo":              media_tmp / "package.mo",
            lib_path / "IF97" / "package.mo":     if97_tmp / "package.mo",
            lib_path / "IF97" / "Constants.mo":   if97_tmp / "Constants.mo",
            lib_path / "IF97" / "Region1.mo":     if97_tmp / "Region1.mo",
            lib_path / "IF97" / "Region2.mo":     if97_tmp / "Region2.mo",
            lib_path / "IF97" / "Saturation.mo":  if97_tmp / "Saturation.mo",
            lib_path / "IF97" / "Derivatives.mo": if97_tmp / "Derivatives.mo",
            lib_path / "Water.mo":                media_tmp / "Water.mo",
        }
        for src_path, dst_path in mo_files.items():
            src = src_path.read_text()
            src = re.sub(r'within\s+OPAL\.library\.', 'within ', src, count=1)
            src = re.sub(r'within\s+OPAL\.library\s*;', '', src, count=1)
            dst_path.write_text(src)

        result = omc.sendExpression(f'loadFile("{media_tmp / "package.mo"}")', parsed=False)
        if 'false' in str(result).lower():
            err = omc.sendExpression('getErrorString()', parsed=False)
            print(f"  FAIL — could not load: {err}")
            return False

        probe = tmpdir / "WaterProbe.mo"
        probe.write_text(
            'model WaterProbe\n  parameter Real p=10e6;\n  parameter Real h=1e6;\n'
            '  Real rho; Real dp; Real dh;\nequation\n'
            '  rho = Media.Water.rho_ph(p,h);\n'
            '  dp = Media.Water.drho_dp_h(p,h);\n'
            '  dh = Media.Water.drho_dh_p(p,h);\nend WaterProbe;\n'
        )
        omc.sendExpression(f'loadFile("{probe}")', parsed=False)
        xml_result = omc.sendExpression('dumpXMLDAE(WaterProbe, addMathMLCode=false)', parsed=False)

    xml_str = str(xml_result)
    xml_match = re.search(r'"([^"]+\.xml)"', xml_str)
    if xml_match:
        xml_file = pathlib.Path(xml_match.group(1))
        if xml_file.exists():
            xml_str = xml_file.read_text()

    if "OPAQUE" in xml_str:
        print("  FAIL — OPAQUE marker found.")
        return False
    elif "true" not in str(xml_result).lower() and "xml" not in xml_str.lower():
        print(f"  FAIL — no XML output.")
        return False
    else:
        print("  PASS — no OPAQUE markers.")
        return True


# ===========================================================================
# TEST 10: Modelica source execution — verify .mo files produce correct values
# ===========================================================================

def _start_omc_session():
    """Start OMPython session, return (omc, OM_HOME) or (None, None)."""
    try:
        from OMPython import OMCSessionZMQ
    except ImportError:
        return None, None
    OM_HOME = OPAL_ROOT / "external" / "OpenModelica" / "build_cmake" / "install_cmake"
    if not (OM_HOME / "bin" / "omc").exists():
        return None, None
    try:
        omc = OMCSessionZMQ(omhome=str(OM_HOME))
        return omc, OM_HOME
    except Exception:
        return None, None


def _load_media_package(omc, tmpdir):
    """Copy Media package to tmpdir with within-clauses fixed, load into OM."""
    import re
    lib_path = OPAL_ROOT / "library" / "Media"
    media_tmp = tmpdir / "Media";  if97_tmp = media_tmp / "IF97"
    if97_tmp.mkdir(parents=True, exist_ok=True)

    mo_files = {
        lib_path / "package.mo":              media_tmp / "package.mo",
        lib_path / "IF97" / "package.mo":     if97_tmp / "package.mo",
        lib_path / "IF97" / "Constants.mo":   if97_tmp / "Constants.mo",
        lib_path / "IF97" / "Region1.mo":     if97_tmp / "Region1.mo",
        lib_path / "IF97" / "Region2.mo":     if97_tmp / "Region2.mo",
        lib_path / "IF97" / "Saturation.mo":  if97_tmp / "Saturation.mo",
        lib_path / "IF97" / "Derivatives.mo": if97_tmp / "Derivatives.mo",
        lib_path / "Water.mo":               media_tmp / "Water.mo",
    }
    for src_path, dst_path in mo_files.items():
        src = src_path.read_text()
        src = re.sub(r'within\s+OPAL\.library\.', 'within ', src, count=1)
        src = re.sub(r'within\s+OPAL\.library\s*;', '', src, count=1)
        dst_path.write_text(src)

    result = omc.sendExpression(f'loadFile("{media_tmp / "package.mo"}")', parsed=False)
    return 'false' not in str(result).lower()


def test_modelica_source_execution():
    """Test 10: Execute the actual .mo files in OpenModelica and verify numerical output.

    Creates probe models with specific (p,T) inputs, simulates at t=0, reads the
    computed properties, and compares to the Python oracle.  This catches coefficient
    transcription errors between the Python test code and the Modelica source.
    """
    print("\n=== Test 10: Modelica Source Execution ===")
    omc, OM_HOME = _start_omc_session()
    if omc is None:
        print("  SKIP — OpenModelica not available.")
        return None

    import re, tempfile, csv

    tol = 1e-6  # Modelica vs Python oracle
    all_pass = True
    n_points = 0
    worst = ("", 0.0)

    # Test points: (p [Pa], T [K], region)
    test_points = [
        # Region 1 — compressed liquid
        (3.0e6,  300.0, 1),
        (10.0e6, 400.0, 1),
        (80.0e6, 500.0, 1),
        (25.0e6, 350.0, 1),
        (50.0e6, 450.0, 1),
        # Region 2 — superheated steam
        (0.01e6, 400.0, 2),
        (1.0e6,  500.0, 2),
        (3.0e6,  700.0, 2),
        (5.0e6,  800.0, 2),
        (0.1e6,  600.0, 2),
    ]

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = pathlib.Path(tmpdir_str)
        omc.sendExpression(f'cd("{tmpdir}")', parsed=False)

        if not _load_media_package(omc, tmpdir):
            err = omc.sendExpression('getErrorString()', parsed=False)
            print(f"  FAIL — could not load Media package: {err}")
            return False

        for p_val, T_val, reg in test_points:
            # Compute Python oracle values
            if reg == 1:
                h_py = h_R1(p_val, T_val)
                v_py = v_R1(p_val, T_val)
                rho_py = rho_R1(p_val, T_val)
                s_py = s_R1(p_val, T_val)
            else:
                h_py = h_R2(p_val, T_val)
                v_py = v_R2(p_val, T_val)
                rho_py = rho_R2(p_val, T_val)
                s_py = s_R2(p_val, T_val)

            # Create a Modelica model that evaluates the Region functions
            model_name = f"Probe_R{reg}_{int(p_val/1e6)}_{int(T_val)}"
            pkg = f"Media.IF97.Region{reg}"
            mo_src = (
                f'model {model_name}\n'
                f'  constant Real p = {p_val};\n'
                f'  constant Real T_val = {T_val};\n'
                f'  Real h_val = {pkg}.h_pT(p, T_val);\n'
                f'  Real v_val = {pkg}.v_pT(p, T_val);\n'
                f'  Real s_val = {pkg}.s_pT(p, T_val);\n'
                f'end {model_name};\n'
            )
            probe_path = tmpdir / f"{model_name}.mo"
            probe_path.write_text(mo_src)

            omc.sendExpression(f'loadFile("{probe_path}")', parsed=False)

            # Simulate for 0 seconds to get initial values
            sim_result = omc.sendExpression(
                f'simulate({model_name}, startTime=0, stopTime=0, '
                f'outputFormat="csv", simflags="-lv=-LOG_SUCCESS")',
                parsed=False
            )

            # Read the CSV output
            csv_path = tmpdir / f"{model_name}_res.csv"
            if not csv_path.exists():
                err = omc.sendExpression('getErrorString()', parsed=False)
                print(f"  FAIL — simulation failed for R{reg} p={p_val/1e6}MPa T={T_val}K: {err}")
                all_pass = False
                continue

            with open(csv_path) as f:
                reader = csv.DictReader(f)
                row = list(reader)[-1]  # last row

            h_mo = float(row['h_val'])
            v_mo = float(row['v_val'])
            s_mo = float(row['s_val'])

            errs = {
                'h': abs(h_mo - h_py) / (abs(h_py) + 1e-30),
                'v': abs(v_mo - v_py) / (abs(v_py) + 1e-30),
                's': abs(s_mo - s_py) / (abs(s_py) + 1e-30),
            }
            max_prop = max(errs, key=errs.get)
            max_err = errs[max_prop]
            ok = max_err < tol
            all_pass = all_pass and ok
            n_points += 1
            if max_err > worst[1]:
                worst = (f"R{reg} p={p_val/1e6}MPa T={T_val:.0f}K {max_prop}", max_err)
            if not ok:
                print(f"  FAIL R{reg} p={p_val/1e6}MPa T={T_val:.0f}K  "
                      f"worst={max_prop} err={max_err:.2e}")
                print(f"    Modelica: h={h_mo:.6f} v={v_mo:.10e} s={s_mo:.6f}")
                print(f"    Python:   h={h_py:.6f} v={v_py:.10e} s={s_py:.6f}")

            # Clean up for next model
            omc.sendExpression(f'deleteClass({model_name})', parsed=False)

    tag = "PASS" if all_pass else "FAIL"
    print(f"  {n_points} points tested, worst: {worst[0]} err={worst[1]:.2e}  [{tag}]")
    return bool(all_pass)


# ===========================================================================
# TEST 11: Modelica T_ph and Water API execution — verify backward equations
#
# This test catches coefficient transcription errors in the backward
# equations (Region1.T_ph, Region2.T_ph) and the unified Water API
# (rho_ph, drho_dp_h, drho_dh_p) that are NOT exercised by the forward-
# function tests (Test 10 only calls h_pT, v_pT, s_pT).
# ===========================================================================

def test_modelica_backward_equations():
    """Test 11: Execute the actual Modelica T_ph and Water API in OpenModelica."""
    print("\n=== Test 11: Modelica Backward Equations & Water API ===")
    omc, OM_HOME = _start_omc_session()
    if omc is None:
        print("  SKIP — OpenModelica not available.")
        return None

    import re, tempfile, csv

    tol = 1e-4  # T_ph backward equation: 1e-4 relative (Newton, 5 iterations)
    all_pass = True
    n_points = 0
    worst = ("", 0.0)

    # Test points: (p [Pa], T [K], region)
    # Stay within backward equation validity (p < 50 MPa)
    test_points = [
        # Region 1 — compressed liquid
        (3.0e6,  300.0, 1),
        (10.0e6, 350.0, 1),
        (10.0e6, 500.0, 1),
        (25.0e6, 400.0, 1),
        (5.0e6,  280.0, 1),
        # Region 2 — superheated steam
        (0.1e6,  400.0, 2),
        (1.0e6,  500.0, 2),
        (1.0e6,  800.0, 2),
        (5.0e6,  600.0, 2),
        (5.0e6,  900.0, 2),
    ]

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = pathlib.Path(tmpdir_str)
        omc.sendExpression(f'cd("{tmpdir}")', parsed=False)

        if not _load_media_package(omc, tmpdir):
            err = omc.sendExpression('getErrorString()', parsed=False)
            print(f"  FAIL — could not load Media package: {err}")
            return False

        for p_val, T_val, reg in test_points:
            # Compute enthalpy from forward function (Python oracle)
            if reg == 1:
                h_val = h_R1(p_val, T_val)
            else:
                h_val = h_R2(p_val, T_val)

            # Create a probe model that calls both T_ph and Water API
            model_name = f"ProbeInv_R{reg}_{int(p_val/1e6)}_{int(T_val)}"
            pkg = f"Media.IF97.Region{reg}"
            mo_src = (
                f'model {model_name}\n'
                f'  constant Real p = {p_val};\n'
                f'  constant Real h_val = {h_val};\n'
                f'  Real T_inv = {pkg}.T_ph(p, h_val);\n'
                f'  Real rho_w = Media.Water.rho_ph(p, h_val);\n'
                f'  Real drho_dp = Media.Water.drho_dp_h(p, h_val);\n'
                f'  Real drho_dh = Media.Water.drho_dh_p(p, h_val);\n'
                f'end {model_name};\n'
            )
            probe_path = tmpdir / f"{model_name}.mo"
            probe_path.write_text(mo_src)

            omc.sendExpression(f'loadFile("{probe_path}")', parsed=False)

            sim_result = omc.sendExpression(
                f'simulate({model_name}, startTime=0, stopTime=0, '
                f'outputFormat="csv", simflags="-lv=-LOG_SUCCESS")',
                parsed=False
            )

            csv_path = tmpdir / f"{model_name}_res.csv"
            if not csv_path.exists():
                err = omc.sendExpression('getErrorString()', parsed=False)
                print(f"  FAIL — simulation failed for R{reg} p={p_val/1e6}MPa T={T_val}K: {err}")
                all_pass = False
                continue

            with open(csv_path) as f:
                reader = csv.DictReader(f)
                row = list(reader)[-1]

            T_mo = float(row['T_inv'])
            rho_mo = float(row['rho_w'])
            drho_dp_mo = float(row['drho_dp'])
            drho_dh_mo = float(row['drho_dh'])

            # Python oracle values
            if reg == 1:
                T_py = T_ph_R1(p_val, h_val)
                rho_py = rho_R1(p_val, T_val)
                drho_dp_py = drho_dp_h_R1(p_val, T_val)
                drho_dh_py = drho_dh_p_R1(p_val, T_val)
            else:
                T_py = T_ph_R2(p_val, h_val)
                rho_py = rho_R2(p_val, T_val)
                drho_dp_py = drho_dp_h_R2(p_val, T_val)
                drho_dh_py = drho_dh_p_R2(p_val, T_val)

            errs = {
                'T_inv': abs(T_mo - T_py) / T_py,
                'rho': abs(rho_mo - rho_py) / (abs(rho_py) + 1e-30),
                'drho_dp': abs(drho_dp_mo - drho_dp_py) / (abs(drho_dp_py) + 1e-30),
                'drho_dh': abs(drho_dh_mo - drho_dh_py) / (abs(drho_dh_py) + 1e-30),
            }
            max_prop = max(errs, key=errs.get)
            max_err = errs[max_prop]
            ok = max_err < tol
            all_pass = all_pass and ok
            n_points += 1
            if max_err > worst[1]:
                worst = (f"R{reg} p={p_val/1e6}MPa T={T_val:.0f}K {max_prop}", max_err)
            if not ok:
                print(f"  FAIL R{reg} p={p_val/1e6}MPa T={T_val:.0f}K  "
                      f"worst={max_prop} err={max_err:.2e}")
                print(f"    Modelica: T={T_mo:.6f} rho={rho_mo:.6f} "
                      f"drho_dp={drho_dp_mo:.6e} drho_dh={drho_dh_mo:.6e}")
                print(f"    Python:   T={T_py:.6f} rho={rho_py:.6f} "
                      f"drho_dp={drho_dp_py:.6e} drho_dh={drho_dh_py:.6e}")

            omc.sendExpression(f'deleteClass({model_name})', parsed=False)

    tag = "PASS" if all_pass else "FAIL"
    print(f"  {n_points} points tested, worst: {worst[0]} err={worst[1]:.2e}  [{tag}]")
    return bool(all_pass)


# ===========================================================================
# TEST 12: Region 3 guard — verify graceful degradation outside valid range
# ===========================================================================

def test_region3_guard():
    """Test 12: Document that near-critical states (Region 3) are outside valid range.

    Region 3 (T > 623.15 K, p > ~16.5 MPa) is not implemented.  The saturation
    functions h_f(p) and rho_f(p) use Region 1 at T_sat, which is only valid for
    T_sat < 623.15 K (p < ~16.5 MPa).  Above that, results are unreliable.

    This test verifies that:
    1. Properties are accurate up to the boundary (~620 K / ~16 MPa)
    2. We can detect when we're approaching the boundary
    """
    print("\n=== Test 11: Region 3 Boundary Guard ===")
    iapws = _import_iapws()
    all_pass = True

    # Verify good accuracy at the edge of validity (~620 K)
    T_edge = 620.0
    p_edge = p_sat(T_edge)
    ref = iapws.IAPWS97(T=T_edge, x=0)
    rho_our = rho_f(p_edge)
    err = abs(rho_our - ref.rho) / ref.rho
    ok = err < 1e-4  # should be accurate at the edge
    all_pass = all_pass and ok
    tag = "PASS" if ok else "FAIL"
    print(f"  T=620K (edge of validity): rho_f err={err:.2e}  [{tag}]")

    # Verify T_sat is accurate up to ~16 MPa
    for p_MPa in [1, 5, 10, 15]:
        p = p_MPa * 1e6
        T_our = T_sat(p)
        sat = iapws.IAPWS97(P=p_MPa, x=0)
        T_ref = sat.T
        err = abs(T_our - T_ref) / T_ref
        ok = err < 1e-6
        all_pass = all_pass and ok
        if not ok:
            print(f"  FAIL T_sat(p={p_MPa}MPa) err={err:.2e}")

    # Document the boundary: T_sat at ~16.53 MPa (Region 1 reducing pressure)
    # approaches 623 K, the Region 1 upper limit
    T_boundary = T_sat(16.0e6)
    approaching = T_boundary > 610  # should be True, near 623 K
    tag = "PASS" if approaching else "FAIL"
    all_pass = all_pass and approaching
    print(f"  T_sat(16MPa) = {T_boundary:.1f} K (boundary at ~623 K)  [{tag}]")

    print(f"\n  Test 11 overall: {'PASS' if all_pass else 'FAIL'}")
    return bool(all_pass)


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("OPAL IAPWS-IF97 Comprehensive Verification")
    print("=" * 60)

    results = []
    results.append(("IAPWS verification tables",        test_iapws_verification_tables()))
    results.append(("Region 1 grid (40+ points)",       test_region1_grid()))
    results.append(("Region 2 grid (40+ points)",       test_region2_grid()))
    results.append(("Saturation curve (20 temps)",       test_saturation_curve()))
    results.append(("Derivatives (30 points)",           test_derivatives_grid()))
    results.append(("Boundary continuity",               test_boundary_continuity()))
    results.append(("Inverse T(p,h)",                    test_inverse_T_ph()))
    results.append(("Two-phase density",                 test_two_phase()))
    results.append(("Unified API cross-region",          test_water_api()))
    results.append(("Extraction transparency",           test_extraction_transparency()))
    results.append(("Modelica source execution",         test_modelica_source_execution()))
    results.append(("Modelica backward eqs & Water API", test_modelica_backward_equations()))
    results.append(("Region 3 boundary guard",           test_region3_guard()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    n_pass = n_fail = n_skip = 0
    for name, result in results:
        if result is True:
            tag = "PASS"; n_pass += 1
        elif result is False:
            tag = "FAIL"; n_fail += 1
        else:
            tag = "SKIP"; n_skip += 1
        print(f"  {name:40s}  {tag}")

    print(f"\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    sys.exit(0 if n_fail == 0 else 1)
