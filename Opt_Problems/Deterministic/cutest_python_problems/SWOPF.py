from Opt_Problems.Deterministic.s2mpjlib import *


class SWOPF(CUTEst_problem):

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    #
    #    Problem:
    #    ********
    #
    #    An optimal electrical powerflow system design problem from Switzerland.
    #
    #    Source:
    #    a contribution to fullfill the LANCELOT academic licence agreement.
    #
    #    SIF input: R. Bacher, Dept of Electrical Engineering, ETH Zurich,
    #               November 1994.
    #
    #    classification = "C-CLQR2-RN-83-92"
    #
    #    Number of nodes       =   7
    #    Number of branches    =   7
    #    Number of generators  =   3
    #
    #
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #   Translated to Python by S2MPJ version 25 XI 2024
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = "SWOPF"

    def __init__(self, *args):
        import numpy as np
        from scipy.sparse import csr_matrix

        nargin = len(args)

        # %%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_ = {}
        ix_ = {}
        ig_ = {}
        v_["FIRST"] = 1
        v_["NOBRANCHES"] = 7
        v_["NOSHUNTS"] = 0
        v_["NOTRAFOS"] = 3
        v_["NOBUSSES"] = 7
        v_["NOGEN"] = 3
        v_["NOGENBK"] = 3
        v_["NOAREAS"] = 0
        # %%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = np.array([])
        self.xscale = np.array([])
        intvars = np.array([])
        binvars = np.array([])
        irA = np.array([], dtype=int)
        icA = np.array([], dtype=int)
        valA = np.array([], dtype=float)
        [iv, ix_, _] = s2mpj_ii("VE0001", ix_)
        self.xnames = arrset(self.xnames, iv, "VE0001")
        [iv, ix_, _] = s2mpj_ii("VF0001", ix_)
        self.xnames = arrset(self.xnames, iv, "VF0001")
        [iv, ix_, _] = s2mpj_ii("V20001", ix_)
        self.xnames = arrset(self.xnames, iv, "V20001")
        [iv, ix_, _] = s2mpj_ii("VE0002", ix_)
        self.xnames = arrset(self.xnames, iv, "VE0002")
        [iv, ix_, _] = s2mpj_ii("VF0002", ix_)
        self.xnames = arrset(self.xnames, iv, "VF0002")
        [iv, ix_, _] = s2mpj_ii("V20002", ix_)
        self.xnames = arrset(self.xnames, iv, "V20002")
        [iv, ix_, _] = s2mpj_ii("VE0003", ix_)
        self.xnames = arrset(self.xnames, iv, "VE0003")
        [iv, ix_, _] = s2mpj_ii("VF0003", ix_)
        self.xnames = arrset(self.xnames, iv, "VF0003")
        [iv, ix_, _] = s2mpj_ii("V20003", ix_)
        self.xnames = arrset(self.xnames, iv, "V20003")
        [iv, ix_, _] = s2mpj_ii("VE0004", ix_)
        self.xnames = arrset(self.xnames, iv, "VE0004")
        [iv, ix_, _] = s2mpj_ii("VF0004", ix_)
        self.xnames = arrset(self.xnames, iv, "VF0004")
        [iv, ix_, _] = s2mpj_ii("V20004", ix_)
        self.xnames = arrset(self.xnames, iv, "V20004")
        [iv, ix_, _] = s2mpj_ii("VE0005", ix_)
        self.xnames = arrset(self.xnames, iv, "VE0005")
        [iv, ix_, _] = s2mpj_ii("VF0005", ix_)
        self.xnames = arrset(self.xnames, iv, "VF0005")
        [iv, ix_, _] = s2mpj_ii("V20005", ix_)
        self.xnames = arrset(self.xnames, iv, "V20005")
        [iv, ix_, _] = s2mpj_ii("VE0006", ix_)
        self.xnames = arrset(self.xnames, iv, "VE0006")
        [iv, ix_, _] = s2mpj_ii("VF0006", ix_)
        self.xnames = arrset(self.xnames, iv, "VF0006")
        [iv, ix_, _] = s2mpj_ii("V20006", ix_)
        self.xnames = arrset(self.xnames, iv, "V20006")
        [iv, ix_, _] = s2mpj_ii("VE0007", ix_)
        self.xnames = arrset(self.xnames, iv, "VE0007")
        [iv, ix_, _] = s2mpj_ii("VF0007", ix_)
        self.xnames = arrset(self.xnames, iv, "VF0007")
        [iv, ix_, _] = s2mpj_ii("V20007", ix_)
        self.xnames = arrset(self.xnames, iv, "V20007")
        [iv, ix_, _] = s2mpj_ii("EI0001", ix_)
        self.xnames = arrset(self.xnames, iv, "EI0001")
        [iv, ix_, _] = s2mpj_ii("FI0001", ix_)
        self.xnames = arrset(self.xnames, iv, "FI0001")
        [iv, ix_, _] = s2mpj_ii("EJ0001", ix_)
        self.xnames = arrset(self.xnames, iv, "EJ0001")
        [iv, ix_, _] = s2mpj_ii("FJ0001", ix_)
        self.xnames = arrset(self.xnames, iv, "FJ0001")
        [iv, ix_, _] = s2mpj_ii("PI0001", ix_)
        self.xnames = arrset(self.xnames, iv, "PI0001")
        [iv, ix_, _] = s2mpj_ii("QI0001", ix_)
        self.xnames = arrset(self.xnames, iv, "QI0001")
        [iv, ix_, _] = s2mpj_ii("PJ0001", ix_)
        self.xnames = arrset(self.xnames, iv, "PJ0001")
        [iv, ix_, _] = s2mpj_ii("QJ0001", ix_)
        self.xnames = arrset(self.xnames, iv, "QJ0001")
        [iv, ix_, _] = s2mpj_ii("EI0002", ix_)
        self.xnames = arrset(self.xnames, iv, "EI0002")
        [iv, ix_, _] = s2mpj_ii("FI0002", ix_)
        self.xnames = arrset(self.xnames, iv, "FI0002")
        [iv, ix_, _] = s2mpj_ii("EJ0002", ix_)
        self.xnames = arrset(self.xnames, iv, "EJ0002")
        [iv, ix_, _] = s2mpj_ii("FJ0002", ix_)
        self.xnames = arrset(self.xnames, iv, "FJ0002")
        [iv, ix_, _] = s2mpj_ii("PI0002", ix_)
        self.xnames = arrset(self.xnames, iv, "PI0002")
        [iv, ix_, _] = s2mpj_ii("QI0002", ix_)
        self.xnames = arrset(self.xnames, iv, "QI0002")
        [iv, ix_, _] = s2mpj_ii("PJ0002", ix_)
        self.xnames = arrset(self.xnames, iv, "PJ0002")
        [iv, ix_, _] = s2mpj_ii("QJ0002", ix_)
        self.xnames = arrset(self.xnames, iv, "QJ0002")
        [iv, ix_, _] = s2mpj_ii("EI0003", ix_)
        self.xnames = arrset(self.xnames, iv, "EI0003")
        [iv, ix_, _] = s2mpj_ii("FI0003", ix_)
        self.xnames = arrset(self.xnames, iv, "FI0003")
        [iv, ix_, _] = s2mpj_ii("EJ0003", ix_)
        self.xnames = arrset(self.xnames, iv, "EJ0003")
        [iv, ix_, _] = s2mpj_ii("FJ0003", ix_)
        self.xnames = arrset(self.xnames, iv, "FJ0003")
        [iv, ix_, _] = s2mpj_ii("PI0003", ix_)
        self.xnames = arrset(self.xnames, iv, "PI0003")
        [iv, ix_, _] = s2mpj_ii("QI0003", ix_)
        self.xnames = arrset(self.xnames, iv, "QI0003")
        [iv, ix_, _] = s2mpj_ii("PJ0003", ix_)
        self.xnames = arrset(self.xnames, iv, "PJ0003")
        [iv, ix_, _] = s2mpj_ii("QJ0003", ix_)
        self.xnames = arrset(self.xnames, iv, "QJ0003")
        [iv, ix_, _] = s2mpj_ii("EI0004", ix_)
        self.xnames = arrset(self.xnames, iv, "EI0004")
        [iv, ix_, _] = s2mpj_ii("FI0004", ix_)
        self.xnames = arrset(self.xnames, iv, "FI0004")
        [iv, ix_, _] = s2mpj_ii("EJ0004", ix_)
        self.xnames = arrset(self.xnames, iv, "EJ0004")
        [iv, ix_, _] = s2mpj_ii("FJ0004", ix_)
        self.xnames = arrset(self.xnames, iv, "FJ0004")
        [iv, ix_, _] = s2mpj_ii("PI0004", ix_)
        self.xnames = arrset(self.xnames, iv, "PI0004")
        [iv, ix_, _] = s2mpj_ii("QI0004", ix_)
        self.xnames = arrset(self.xnames, iv, "QI0004")
        [iv, ix_, _] = s2mpj_ii("PJ0004", ix_)
        self.xnames = arrset(self.xnames, iv, "PJ0004")
        [iv, ix_, _] = s2mpj_ii("QJ0004", ix_)
        self.xnames = arrset(self.xnames, iv, "QJ0004")
        [iv, ix_, _] = s2mpj_ii("EI0005", ix_)
        self.xnames = arrset(self.xnames, iv, "EI0005")
        [iv, ix_, _] = s2mpj_ii("FI0005", ix_)
        self.xnames = arrset(self.xnames, iv, "FI0005")
        [iv, ix_, _] = s2mpj_ii("EJ0005", ix_)
        self.xnames = arrset(self.xnames, iv, "EJ0005")
        [iv, ix_, _] = s2mpj_ii("FJ0005", ix_)
        self.xnames = arrset(self.xnames, iv, "FJ0005")
        [iv, ix_, _] = s2mpj_ii("PI0005", ix_)
        self.xnames = arrset(self.xnames, iv, "PI0005")
        [iv, ix_, _] = s2mpj_ii("QI0005", ix_)
        self.xnames = arrset(self.xnames, iv, "QI0005")
        [iv, ix_, _] = s2mpj_ii("PJ0005", ix_)
        self.xnames = arrset(self.xnames, iv, "PJ0005")
        [iv, ix_, _] = s2mpj_ii("QJ0005", ix_)
        self.xnames = arrset(self.xnames, iv, "QJ0005")
        [iv, ix_, _] = s2mpj_ii("EI0006", ix_)
        self.xnames = arrset(self.xnames, iv, "EI0006")
        [iv, ix_, _] = s2mpj_ii("FI0006", ix_)
        self.xnames = arrset(self.xnames, iv, "FI0006")
        [iv, ix_, _] = s2mpj_ii("EJ0006", ix_)
        self.xnames = arrset(self.xnames, iv, "EJ0006")
        [iv, ix_, _] = s2mpj_ii("FJ0006", ix_)
        self.xnames = arrset(self.xnames, iv, "FJ0006")
        [iv, ix_, _] = s2mpj_ii("PI0006", ix_)
        self.xnames = arrset(self.xnames, iv, "PI0006")
        [iv, ix_, _] = s2mpj_ii("QI0006", ix_)
        self.xnames = arrset(self.xnames, iv, "QI0006")
        [iv, ix_, _] = s2mpj_ii("PJ0006", ix_)
        self.xnames = arrset(self.xnames, iv, "PJ0006")
        [iv, ix_, _] = s2mpj_ii("QJ0006", ix_)
        self.xnames = arrset(self.xnames, iv, "QJ0006")
        [iv, ix_, _] = s2mpj_ii("EI0007", ix_)
        self.xnames = arrset(self.xnames, iv, "EI0007")
        [iv, ix_, _] = s2mpj_ii("FI0007", ix_)
        self.xnames = arrset(self.xnames, iv, "FI0007")
        [iv, ix_, _] = s2mpj_ii("EJ0007", ix_)
        self.xnames = arrset(self.xnames, iv, "EJ0007")
        [iv, ix_, _] = s2mpj_ii("FJ0007", ix_)
        self.xnames = arrset(self.xnames, iv, "FJ0007")
        [iv, ix_, _] = s2mpj_ii("PI0007", ix_)
        self.xnames = arrset(self.xnames, iv, "PI0007")
        [iv, ix_, _] = s2mpj_ii("QI0007", ix_)
        self.xnames = arrset(self.xnames, iv, "QI0007")
        [iv, ix_, _] = s2mpj_ii("PJ0007", ix_)
        self.xnames = arrset(self.xnames, iv, "PJ0007")
        [iv, ix_, _] = s2mpj_ii("QJ0007", ix_)
        self.xnames = arrset(self.xnames, iv, "QJ0007")
        [iv, ix_, _] = s2mpj_ii("PG0001", ix_)
        self.xnames = arrset(self.xnames, iv, "PG0001")
        [iv, ix_, _] = s2mpj_ii("PG0002", ix_)
        self.xnames = arrset(self.xnames, iv, "PG0002")
        [iv, ix_, _] = s2mpj_ii("PG0003", ix_)
        self.xnames = arrset(self.xnames, iv, "PG0003")
        [iv, ix_, _] = s2mpj_ii("QG0001", ix_)
        self.xnames = arrset(self.xnames, iv, "QG0001")
        [iv, ix_, _] = s2mpj_ii("QG0002", ix_)
        self.xnames = arrset(self.xnames, iv, "QG0002")
        [iv, ix_, _] = s2mpj_ii("QG0003", ix_)
        self.xnames = arrset(self.xnames, iv, "QG0003")
        # %%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale = np.array([])
        self.grnames = np.array([])
        cnames = np.array([])
        self.cnames = np.array([])
        gtype = np.array([])
        [ig, ig_, _] = s2mpj_ii("GV20001", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GV20001")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["V20001"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("SLF0000", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "SLF0000")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0001"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GV20002", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GV20002")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["V20002"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GV20003", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GV20003")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["V20003"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GV20004", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GV20004")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["V20004"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GV20005", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GV20005")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["V20005"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GV20006", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GV20006")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["V20006"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GV20007", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GV20007")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["V20007"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("LOSS0000", ig_)
        gtype = arrset(gtype, ig, "<>")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PI0001"]])
        valA = np.append(valA, float(1.000))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PJ0001"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GEI0001", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEI0001")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["EI0001"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GFI0001", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFI0001")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["FI0001"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GEJ0001", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEJ0001")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["EJ0001"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GFJ0001", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFJ0001")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["FJ0001"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GEI0001", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEI0001")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0001"]])
        valA = np.append(valA, float(-5.299))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0001"]])
        valA = np.append(valA, float(-66.243))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0002"]])
        valA = np.append(valA, float(5.299))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0002"]])
        valA = np.append(valA, float(66.243))
        [ig, ig_, _] = s2mpj_ii("GFI0001", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFI0001")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0001"]])
        valA = np.append(valA, float(66.243))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0001"]])
        valA = np.append(valA, float(-5.299))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0002"]])
        valA = np.append(valA, float(-66.243))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0002"]])
        valA = np.append(valA, float(5.299))
        [ig, ig_, _] = s2mpj_ii("GEJ0001", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEJ0001")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0002"]])
        valA = np.append(valA, float(-5.299))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0002"]])
        valA = np.append(valA, float(-66.243))
        [ig, ig_, _] = s2mpj_ii("GFJ0001", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFJ0001")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0002"]])
        valA = np.append(valA, float(66.243))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0002"]])
        valA = np.append(valA, float(-5.299))
        [ig, ig_, _] = s2mpj_ii("GEJ0001", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEJ0001")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0001"]])
        valA = np.append(valA, float(5.299))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0001"]])
        valA = np.append(valA, float(66.243))
        [ig, ig_, _] = s2mpj_ii("GFJ0001", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFJ0001")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0001"]])
        valA = np.append(valA, float(-66.243))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0001"]])
        valA = np.append(valA, float(5.299))
        [ig, ig_, _] = s2mpj_ii("GPI0001", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPI0001")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PI0001"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GQI0001", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQI0001")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QI0001"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GPJ0001", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPJ0001")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PJ0001"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GQJ0001", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQJ0001")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QJ0001"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GPNI0001", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPNI0001")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PI0001"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GPNI0002", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPNI0002")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PJ0001"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GQNI0001", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQNI0001")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QI0001"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GQNI0002", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQNI0002")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QJ0001"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GMXI0001", ig_)
        gtype = arrset(gtype, ig, "<=")
        cnames = arrset(cnames, ig, "GMXI0001")
        [ig, ig_, _] = s2mpj_ii("GMXJ0001", ig_)
        gtype = arrset(gtype, ig, "<=")
        cnames = arrset(cnames, ig, "GMXJ0001")
        [ig, ig_, _] = s2mpj_ii("LOSS0000", ig_)
        gtype = arrset(gtype, ig, "<>")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PI0002"]])
        valA = np.append(valA, float(1.000))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PJ0002"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GEI0002", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEI0002")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["EI0002"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GFI0002", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFI0002")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["FI0002"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GEJ0002", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEJ0002")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["EJ0002"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GFJ0002", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFJ0002")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["FJ0002"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GEI0002", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEI0002")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0002"]])
        valA = np.append(valA, float(-1.175))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0002"]])
        valA = np.append(valA, float(-6.915))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0006"]])
        valA = np.append(valA, float(1.175))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0006"]])
        valA = np.append(valA, float(7.051))
        [ig, ig_, _] = s2mpj_ii("GFI0002", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFI0002")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0002"]])
        valA = np.append(valA, float(6.915))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0002"]])
        valA = np.append(valA, float(-1.175))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0006"]])
        valA = np.append(valA, float(-7.051))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0006"]])
        valA = np.append(valA, float(1.175))
        [ig, ig_, _] = s2mpj_ii("GEJ0002", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEJ0002")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0006"]])
        valA = np.append(valA, float(-1.175))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0006"]])
        valA = np.append(valA, float(-6.915))
        [ig, ig_, _] = s2mpj_ii("GFJ0002", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFJ0002")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0006"]])
        valA = np.append(valA, float(6.915))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0006"]])
        valA = np.append(valA, float(-1.175))
        [ig, ig_, _] = s2mpj_ii("GEJ0002", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEJ0002")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0002"]])
        valA = np.append(valA, float(1.175))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0002"]])
        valA = np.append(valA, float(7.051))
        [ig, ig_, _] = s2mpj_ii("GFJ0002", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFJ0002")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0002"]])
        valA = np.append(valA, float(-7.051))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0002"]])
        valA = np.append(valA, float(1.175))
        [ig, ig_, _] = s2mpj_ii("GPI0002", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPI0002")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PI0002"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GQI0002", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQI0002")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QI0002"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GPJ0002", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPJ0002")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PJ0002"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GQJ0002", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQJ0002")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QJ0002"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GPNI0002", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPNI0002")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PI0002"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GPNI0006", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPNI0006")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PJ0002"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GQNI0002", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQNI0002")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QI0002"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GQNI0006", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQNI0006")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QJ0002"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GMXI0002", ig_)
        gtype = arrset(gtype, ig, "<=")
        cnames = arrset(cnames, ig, "GMXI0002")
        [ig, ig_, _] = s2mpj_ii("GMXJ0002", ig_)
        gtype = arrset(gtype, ig, "<=")
        cnames = arrset(cnames, ig, "GMXJ0002")
        [ig, ig_, _] = s2mpj_ii("LOSS0000", ig_)
        gtype = arrset(gtype, ig, "<>")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PI0003"]])
        valA = np.append(valA, float(1.000))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PJ0003"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GEI0003", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEI0003")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["EI0003"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GFI0003", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFI0003")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["FI0003"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GEJ0003", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEJ0003")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["EJ0003"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GFJ0003", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFJ0003")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["FJ0003"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GEI0003", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEI0003")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0002"]])
        valA = np.append(valA, float(-1.726))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0002"]])
        valA = np.append(valA, float(-10.498))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0004"]])
        valA = np.append(valA, float(1.726))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0004"]])
        valA = np.append(valA, float(10.588))
        [ig, ig_, _] = s2mpj_ii("GFI0003", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFI0003")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0002"]])
        valA = np.append(valA, float(10.498))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0002"]])
        valA = np.append(valA, float(-1.726))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0004"]])
        valA = np.append(valA, float(-10.588))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0004"]])
        valA = np.append(valA, float(1.726))
        [ig, ig_, _] = s2mpj_ii("GEJ0003", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEJ0003")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0004"]])
        valA = np.append(valA, float(-1.726))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0004"]])
        valA = np.append(valA, float(-10.498))
        [ig, ig_, _] = s2mpj_ii("GFJ0003", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFJ0003")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0004"]])
        valA = np.append(valA, float(10.498))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0004"]])
        valA = np.append(valA, float(-1.726))
        [ig, ig_, _] = s2mpj_ii("GEJ0003", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEJ0003")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0002"]])
        valA = np.append(valA, float(1.726))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0002"]])
        valA = np.append(valA, float(10.588))
        [ig, ig_, _] = s2mpj_ii("GFJ0003", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFJ0003")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0002"]])
        valA = np.append(valA, float(-10.588))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0002"]])
        valA = np.append(valA, float(1.726))
        [ig, ig_, _] = s2mpj_ii("GPI0003", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPI0003")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PI0003"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GQI0003", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQI0003")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QI0003"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GPJ0003", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPJ0003")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PJ0003"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GQJ0003", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQJ0003")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QJ0003"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GPNI0002", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPNI0002")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PI0003"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GPNI0004", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPNI0004")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PJ0003"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GQNI0002", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQNI0002")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QI0003"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GQNI0004", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQNI0004")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QJ0003"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GMXI0003", ig_)
        gtype = arrset(gtype, ig, "<=")
        cnames = arrset(cnames, ig, "GMXI0003")
        [ig, ig_, _] = s2mpj_ii("GMXJ0003", ig_)
        gtype = arrset(gtype, ig, "<=")
        cnames = arrset(cnames, ig, "GMXJ0003")
        [ig, ig_, _] = s2mpj_ii("LOSS0000", ig_)
        gtype = arrset(gtype, ig, "<>")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PI0004"]])
        valA = np.append(valA, float(1.000))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PJ0004"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GEI0004", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEI0004")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["EI0004"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GFI0004", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFI0004")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["FI0004"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GEJ0004", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEJ0004")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["EJ0004"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GFJ0004", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFJ0004")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["FJ0004"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GEI0004", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEI0004")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0003"]])
        valA = np.append(valA, float(-6.897))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0003"]])
        valA = np.append(valA, float(-82.759))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0004"]])
        valA = np.append(valA, float(6.897))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0004"]])
        valA = np.append(valA, float(82.759))
        [ig, ig_, _] = s2mpj_ii("GFI0004", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFI0004")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0003"]])
        valA = np.append(valA, float(82.759))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0003"]])
        valA = np.append(valA, float(-6.897))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0004"]])
        valA = np.append(valA, float(-82.759))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0004"]])
        valA = np.append(valA, float(6.897))
        [ig, ig_, _] = s2mpj_ii("GEJ0004", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEJ0004")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0004"]])
        valA = np.append(valA, float(-6.897))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0004"]])
        valA = np.append(valA, float(-82.759))
        [ig, ig_, _] = s2mpj_ii("GFJ0004", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFJ0004")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0004"]])
        valA = np.append(valA, float(82.759))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0004"]])
        valA = np.append(valA, float(-6.897))
        [ig, ig_, _] = s2mpj_ii("GEJ0004", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEJ0004")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0003"]])
        valA = np.append(valA, float(6.897))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0003"]])
        valA = np.append(valA, float(82.759))
        [ig, ig_, _] = s2mpj_ii("GFJ0004", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFJ0004")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0003"]])
        valA = np.append(valA, float(-82.759))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0003"]])
        valA = np.append(valA, float(6.897))
        [ig, ig_, _] = s2mpj_ii("GPI0004", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPI0004")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PI0004"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GQI0004", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQI0004")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QI0004"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GPJ0004", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPJ0004")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PJ0004"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GQJ0004", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQJ0004")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QJ0004"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GPNI0003", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPNI0003")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PI0004"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GPNI0004", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPNI0004")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PJ0004"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GQNI0003", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQNI0003")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QI0004"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GQNI0004", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQNI0004")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QJ0004"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GMXI0004", ig_)
        gtype = arrset(gtype, ig, "<=")
        cnames = arrset(cnames, ig, "GMXI0004")
        [ig, ig_, _] = s2mpj_ii("GMXJ0004", ig_)
        gtype = arrset(gtype, ig, "<=")
        cnames = arrset(cnames, ig, "GMXJ0004")
        [ig, ig_, _] = s2mpj_ii("LOSS0000", ig_)
        gtype = arrset(gtype, ig, "<>")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PI0005"]])
        valA = np.append(valA, float(1.000))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PJ0005"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GEI0005", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEI0005")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["EI0005"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GFI0005", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFI0005")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["FI0005"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GEJ0005", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEJ0005")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["EJ0005"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GFJ0005", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFJ0005")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["FJ0005"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GEI0005", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEI0005")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0004"]])
        valA = np.append(valA, float(-1.175))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0004"]])
        valA = np.append(valA, float(-6.915))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0007"]])
        valA = np.append(valA, float(1.175))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0007"]])
        valA = np.append(valA, float(7.051))
        [ig, ig_, _] = s2mpj_ii("GFI0005", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFI0005")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0004"]])
        valA = np.append(valA, float(6.915))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0004"]])
        valA = np.append(valA, float(-1.175))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0007"]])
        valA = np.append(valA, float(-7.051))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0007"]])
        valA = np.append(valA, float(1.175))
        [ig, ig_, _] = s2mpj_ii("GEJ0005", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEJ0005")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0007"]])
        valA = np.append(valA, float(-1.175))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0007"]])
        valA = np.append(valA, float(-6.915))
        [ig, ig_, _] = s2mpj_ii("GFJ0005", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFJ0005")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0007"]])
        valA = np.append(valA, float(6.915))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0007"]])
        valA = np.append(valA, float(-1.175))
        [ig, ig_, _] = s2mpj_ii("GEJ0005", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEJ0005")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0004"]])
        valA = np.append(valA, float(1.175))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0004"]])
        valA = np.append(valA, float(7.051))
        [ig, ig_, _] = s2mpj_ii("GFJ0005", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFJ0005")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0004"]])
        valA = np.append(valA, float(-7.051))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0004"]])
        valA = np.append(valA, float(1.175))
        [ig, ig_, _] = s2mpj_ii("GPI0005", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPI0005")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PI0005"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GQI0005", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQI0005")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QI0005"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GPJ0005", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPJ0005")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PJ0005"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GQJ0005", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQJ0005")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QJ0005"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GPNI0004", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPNI0004")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PI0005"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GPNI0007", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPNI0007")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PJ0005"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GQNI0004", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQNI0004")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QI0005"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GQNI0007", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQNI0007")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QJ0005"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GMXI0005", ig_)
        gtype = arrset(gtype, ig, "<=")
        cnames = arrset(cnames, ig, "GMXI0005")
        [ig, ig_, _] = s2mpj_ii("GMXJ0005", ig_)
        gtype = arrset(gtype, ig, "<=")
        cnames = arrset(cnames, ig, "GMXJ0005")
        [ig, ig_, _] = s2mpj_ii("LOSS0000", ig_)
        gtype = arrset(gtype, ig, "<>")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PI0006"]])
        valA = np.append(valA, float(1.000))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PJ0006"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GEI0006", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEI0006")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["EI0006"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GFI0006", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFI0006")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["FI0006"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GEJ0006", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEJ0006")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["EJ0006"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GFJ0006", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFJ0006")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["FJ0006"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GEI0006", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEI0006")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0005"]])
        valA = np.append(valA, float(-3.448))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0005"]])
        valA = np.append(valA, float(-41.379))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0006"]])
        valA = np.append(valA, float(3.448))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0006"]])
        valA = np.append(valA, float(41.379))
        [ig, ig_, _] = s2mpj_ii("GFI0006", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFI0006")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0005"]])
        valA = np.append(valA, float(41.379))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0005"]])
        valA = np.append(valA, float(-3.448))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0006"]])
        valA = np.append(valA, float(-41.379))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0006"]])
        valA = np.append(valA, float(3.448))
        [ig, ig_, _] = s2mpj_ii("GEJ0006", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEJ0006")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0006"]])
        valA = np.append(valA, float(-3.448))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0006"]])
        valA = np.append(valA, float(-41.379))
        [ig, ig_, _] = s2mpj_ii("GFJ0006", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFJ0006")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0006"]])
        valA = np.append(valA, float(41.379))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0006"]])
        valA = np.append(valA, float(-3.448))
        [ig, ig_, _] = s2mpj_ii("GEJ0006", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEJ0006")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0005"]])
        valA = np.append(valA, float(3.448))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0005"]])
        valA = np.append(valA, float(41.379))
        [ig, ig_, _] = s2mpj_ii("GFJ0006", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFJ0006")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0005"]])
        valA = np.append(valA, float(-41.379))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0005"]])
        valA = np.append(valA, float(3.448))
        [ig, ig_, _] = s2mpj_ii("GPI0006", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPI0006")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PI0006"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GQI0006", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQI0006")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QI0006"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GPJ0006", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPJ0006")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PJ0006"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GQJ0006", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQJ0006")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QJ0006"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GPNI0005", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPNI0005")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PI0006"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GPNI0006", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPNI0006")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PJ0006"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GQNI0005", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQNI0005")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QI0006"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GQNI0006", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQNI0006")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QJ0006"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GMXI0006", ig_)
        gtype = arrset(gtype, ig, "<=")
        cnames = arrset(cnames, ig, "GMXI0006")
        [ig, ig_, _] = s2mpj_ii("GMXJ0006", ig_)
        gtype = arrset(gtype, ig, "<=")
        cnames = arrset(cnames, ig, "GMXJ0006")
        [ig, ig_, _] = s2mpj_ii("LOSS0000", ig_)
        gtype = arrset(gtype, ig, "<>")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PI0007"]])
        valA = np.append(valA, float(1.000))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PJ0007"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GEI0007", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEI0007")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["EI0007"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GFI0007", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFI0007")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["FI0007"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GEJ0007", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEJ0007")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["EJ0007"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GFJ0007", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFJ0007")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["FJ0007"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GEI0007", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEI0007")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0006"]])
        valA = np.append(valA, float(-1.726))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0006"]])
        valA = np.append(valA, float(-10.498))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0007"]])
        valA = np.append(valA, float(1.726))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0007"]])
        valA = np.append(valA, float(10.588))
        [ig, ig_, _] = s2mpj_ii("GFI0007", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFI0007")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0006"]])
        valA = np.append(valA, float(10.498))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0006"]])
        valA = np.append(valA, float(-1.726))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0007"]])
        valA = np.append(valA, float(-10.588))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0007"]])
        valA = np.append(valA, float(1.726))
        [ig, ig_, _] = s2mpj_ii("GEJ0007", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEJ0007")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0007"]])
        valA = np.append(valA, float(-1.726))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0007"]])
        valA = np.append(valA, float(-10.498))
        [ig, ig_, _] = s2mpj_ii("GFJ0007", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFJ0007")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0007"]])
        valA = np.append(valA, float(10.498))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0007"]])
        valA = np.append(valA, float(-1.726))
        [ig, ig_, _] = s2mpj_ii("GEJ0007", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GEJ0007")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0006"]])
        valA = np.append(valA, float(1.726))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0006"]])
        valA = np.append(valA, float(10.588))
        [ig, ig_, _] = s2mpj_ii("GFJ0007", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GFJ0007")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VE0006"]])
        valA = np.append(valA, float(-10.588))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["VF0006"]])
        valA = np.append(valA, float(1.726))
        [ig, ig_, _] = s2mpj_ii("GPI0007", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPI0007")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PI0007"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GQI0007", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQI0007")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QI0007"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GPJ0007", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPJ0007")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PJ0007"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GQJ0007", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQJ0007")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QJ0007"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GPNI0006", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPNI0006")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PI0007"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GPNI0007", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPNI0007")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PJ0007"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GQNI0006", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQNI0006")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QI0007"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GQNI0007", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQNI0007")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QJ0007"]])
        valA = np.append(valA, float(-1.000))
        [ig, ig_, _] = s2mpj_ii("GMXI0007", ig_)
        gtype = arrset(gtype, ig, "<=")
        cnames = arrset(cnames, ig, "GMXI0007")
        [ig, ig_, _] = s2mpj_ii("GMXJ0007", ig_)
        gtype = arrset(gtype, ig, "<=")
        cnames = arrset(cnames, ig, "GMXJ0007")
        [ig, ig_, _] = s2mpj_ii("GPNI0001", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPNI0001")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PG0001"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GPNI0003", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPNI0003")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PG0002"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GPNI0005", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GPNI0005")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["PG0003"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GQNI0001", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQNI0001")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QG0001"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GQNI0003", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQNI0003")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QG0002"]])
        valA = np.append(valA, float(1.000))
        [ig, ig_, _] = s2mpj_ii("GQNI0005", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "GQNI0005")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["QG0003"]])
        valA = np.append(valA, float(1.000))
        # %%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n = len(ix_)
        ngrp = len(ig_)
        legrps = np.where(gtype == "<=")[0]
        eqgrps = np.where(gtype == "==")[0]
        gegrps = np.where(gtype == ">=")[0]
        self.nle = len(legrps)
        self.neq = len(eqgrps)
        self.nge = len(gegrps)
        self.m = self.nle + self.neq + self.nge
        self.congrps = np.concatenate((legrps, eqgrps, gegrps))
        self.cnames = cnames[self.congrps]
        self.nob = ngrp - self.m
        self.objgrps = np.where(gtype == "<>")[0]
        # %%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = np.zeros((ngrp, 1))
        self.gconst = arrset(self.gconst, ig_["GPNI0001"], float(0.000))
        self.gconst = arrset(self.gconst, ig_["GQNI0001"], float(0.000))
        self.gconst = arrset(self.gconst, ig_["SLF0000"], float(0.000))
        self.gconst = arrset(self.gconst, ig_["GPNI0002"], float(2.000))
        self.gconst = arrset(self.gconst, ig_["GQNI0002"], float(3.000))
        self.gconst = arrset(self.gconst, ig_["GPNI0003"], float(0.600))
        self.gconst = arrset(self.gconst, ig_["GQNI0003"], float(0.080))
        self.gconst = arrset(self.gconst, ig_["GPNI0004"], float(2.000))
        self.gconst = arrset(self.gconst, ig_["GQNI0004"], float(0.200))
        self.gconst = arrset(self.gconst, ig_["GPNI0005"], float(0.500))
        self.gconst = arrset(self.gconst, ig_["GQNI0005"], float(0.050))
        self.gconst = arrset(self.gconst, ig_["GPNI0006"], float(1.000))
        self.gconst = arrset(self.gconst, ig_["GQNI0006"], float(0.300))
        self.gconst = arrset(self.gconst, ig_["GPNI0007"], float(2.000))
        self.gconst = arrset(self.gconst, ig_["GQNI0007"], float(1.000))
        self.gconst = arrset(self.gconst, ig_["GMXI0001"], float(16.000))
        self.gconst = arrset(self.gconst, ig_["GMXJ0001"], float(16.000))
        self.gconst = arrset(self.gconst, ig_["GMXI0002"], float(4.000))
        self.gconst = arrset(self.gconst, ig_["GMXJ0002"], float(4.000))
        self.gconst = arrset(self.gconst, ig_["GMXI0003"], float(4.000))
        self.gconst = arrset(self.gconst, ig_["GMXJ0003"], float(4.000))
        self.gconst = arrset(self.gconst, ig_["GMXI0004"], float(25.000))
        self.gconst = arrset(self.gconst, ig_["GMXJ0004"], float(25.000))
        self.gconst = arrset(self.gconst, ig_["GMXI0005"], float(4.000))
        self.gconst = arrset(self.gconst, ig_["GMXJ0005"], float(4.000))
        self.gconst = arrset(self.gconst, ig_["GMXI0006"], float(6.250))
        self.gconst = arrset(self.gconst, ig_["GMXJ0006"], float(6.250))
        self.gconst = arrset(self.gconst, ig_["GMXI0007"], float(4.000))
        self.gconst = arrset(self.gconst, ig_["GMXJ0007"], float(4.000))
        # %%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = np.full((self.n, 1), -float("Inf"))
        self.xupper = np.full((self.n, 1), +float("Inf"))
        self.xlower[ix_["V20001"]] = 0.810
        self.xupper[ix_["V20001"]] = 1.210
        self.xlower[ix_["V20002"]] = 0.810
        self.xupper[ix_["V20002"]] = 1.210
        self.xlower[ix_["V20003"]] = 0.941
        self.xupper[ix_["V20003"]] = 1.210
        self.xlower[ix_["V20004"]] = 0.941
        self.xupper[ix_["V20004"]] = 1.210
        self.xlower[ix_["V20005"]] = 0.941
        self.xupper[ix_["V20005"]] = 1.210
        self.xlower[ix_["V20006"]] = 0.941
        self.xupper[ix_["V20006"]] = 1.210
        self.xlower[ix_["V20007"]] = 0.941
        self.xupper[ix_["V20007"]] = 1.210
        self.xlower[ix_["PG0001"]] = 0.500
        self.xupper[ix_["PG0001"]] = 10.000
        self.xlower[ix_["PG0002"]] = 0.500
        self.xupper[ix_["PG0002"]] = 10.000
        self.xlower[ix_["PG0003"]] = 0.200
        self.xupper[ix_["PG0003"]] = 4.000
        # %%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = np.zeros((self.n, 1))
        self.y0 = np.zeros((self.m, 1))
        self.x0[ix_["VE0001"]] = float(1.000)
        self.x0[ix_["VF0001"]] = float(0.000)
        self.x0[ix_["V20001"]] = float(1.000)
        self.x0[ix_["VE0002"]] = float(1.001)
        self.x0[ix_["VF0002"]] = float(0.000)
        self.x0[ix_["V20002"]] = float(1.002)
        self.x0[ix_["VE0003"]] = float(1.050)
        self.x0[ix_["VF0003"]] = float(0.000)
        self.x0[ix_["V20003"]] = float(1.102)
        self.x0[ix_["VE0004"]] = float(1.001)
        self.x0[ix_["VF0004"]] = float(0.000)
        self.x0[ix_["V20004"]] = float(1.002)
        self.x0[ix_["VE0005"]] = float(1.050)
        self.x0[ix_["VF0005"]] = float(0.000)
        self.x0[ix_["V20005"]] = float(1.102)
        self.x0[ix_["VE0006"]] = float(1.001)
        self.x0[ix_["VF0006"]] = float(0.000)
        self.x0[ix_["V20006"]] = float(1.002)
        self.x0[ix_["VE0007"]] = float(1.001)
        self.x0[ix_["VF0007"]] = float(0.000)
        self.x0[ix_["V20007"]] = float(1.002)
        self.x0[ix_["EI0001"]] = float(-0.005)
        self.x0[ix_["FI0001"]] = float(0.066)
        self.x0[ix_["EJ0001"]] = float(0.005)
        self.x0[ix_["FJ0001"]] = float(-0.066)
        self.x0[ix_["PI0001"]] = float(-0.005)
        self.x0[ix_["QI0001"]] = float(-0.066)
        self.x0[ix_["PJ0001"]] = float(0.005)
        self.x0[ix_["QJ0001"]] = float(0.066)
        self.x0[ix_["EI0002"]] = float(0.000)
        self.x0[ix_["FI0002"]] = float(0.136)
        self.x0[ix_["EJ0002"]] = float(0.000)
        self.x0[ix_["FJ0002"]] = float(0.136)
        self.x0[ix_["PI0002"]] = float(0.000)
        self.x0[ix_["QI0002"]] = float(-0.136)
        self.x0[ix_["PJ0002"]] = float(0.000)
        self.x0[ix_["QJ0002"]] = float(-0.136)
        self.x0[ix_["EI0003"]] = float(0.000)
        self.x0[ix_["FI0003"]] = float(0.091)
        self.x0[ix_["EJ0003"]] = float(0.000)
        self.x0[ix_["FJ0003"]] = float(0.091)
        self.x0[ix_["PI0003"]] = float(0.000)
        self.x0[ix_["QI0003"]] = float(-0.091)
        self.x0[ix_["PJ0003"]] = float(0.000)
        self.x0[ix_["QJ0003"]] = float(-0.091)
        self.x0[ix_["EI0004"]] = float(0.338)
        self.x0[ix_["FI0004"]] = float(-4.055)
        self.x0[ix_["EJ0004"]] = float(-0.338)
        self.x0[ix_["FJ0004"]] = float(4.055)
        self.x0[ix_["PI0004"]] = float(0.355)
        self.x0[ix_["QI0004"]] = float(4.258)
        self.x0[ix_["PJ0004"]] = float(-0.338)
        self.x0[ix_["QJ0004"]] = float(-4.059)
        self.x0[ix_["EI0005"]] = float(0.000)
        self.x0[ix_["FI0005"]] = float(0.136)
        self.x0[ix_["EJ0005"]] = float(0.000)
        self.x0[ix_["FJ0005"]] = float(0.136)
        self.x0[ix_["PI0005"]] = float(0.000)
        self.x0[ix_["QI0005"]] = float(-0.136)
        self.x0[ix_["PJ0005"]] = float(0.000)
        self.x0[ix_["QJ0005"]] = float(-0.136)
        self.x0[ix_["EI0006"]] = float(0.169)
        self.x0[ix_["FI0006"]] = float(-2.028)
        self.x0[ix_["EJ0006"]] = float(-0.169)
        self.x0[ix_["FJ0006"]] = float(2.028)
        self.x0[ix_["PI0006"]] = float(0.177)
        self.x0[ix_["QI0006"]] = float(2.129)
        self.x0[ix_["PJ0006"]] = float(-0.169)
        self.x0[ix_["QJ0006"]] = float(-2.030)
        self.x0[ix_["EI0007"]] = float(0.000)
        self.x0[ix_["FI0007"]] = float(0.091)
        self.x0[ix_["EJ0007"]] = float(0.000)
        self.x0[ix_["FJ0007"]] = float(0.091)
        self.x0[ix_["PI0007"]] = float(0.000)
        self.x0[ix_["QI0007"]] = float(-0.091)
        self.x0[ix_["PJ0007"]] = float(0.000)
        self.x0[ix_["QJ0007"]] = float(-0.091)
        self.x0[ix_["PG0001"]] = float(3.000)
        self.x0[ix_["PG0002"]] = float(5.000)
        self.x0[ix_["PG0003"]] = float(2.000)
        self.x0[ix_["QG0001"]] = float(0.000)
        self.x0[ix_["QG0002"]] = float(0.000)
        self.x0[ix_["QG0003"]] = float(0.000)
        # %%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_ = {}
        elftv = []
        [it, iet_, _] = s2mpj_ii("eXTIMESY", iet_)
        elftv = loaset(elftv, it, 0, "X")
        elftv = loaset(elftv, it, 1, "Y")
        [it, iet_, _] = s2mpj_ii("eXSQUARE", iet_)
        elftv = loaset(elftv, it, 0, "X")
        # %%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = np.array([])
        ielftype = np.array([])
        self.elvar = []
        ename = "E20001"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "VE0001"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "F20001"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "VF0001"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "E20002"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "VE0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "F20002"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "VF0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "E20003"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "VE0003"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "F20003"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "VF0003"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "E20004"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "VE0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "F20004"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "VF0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "E20005"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "VE0005"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "F20005"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "VF0005"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "E20006"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "VE0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "F20006"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "VF0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "E20007"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "VE0007"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "F20007"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "VF0007"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EIEI0001"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EI0001"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0001"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FIFI0001"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FI0001"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0001"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EIFI0001"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EI0001"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0001"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FIEI0001"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FI0001"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0001"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EJEJ0001"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EJ0001"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FJFJ0001"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FJ0001"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EJFJ0001"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EJ0001"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FJEJ0001"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FJ0001"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "PI20001"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "PI0001"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "QI20001"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "QI0001"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "PJ20001"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "PJ0001"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "QJ20001"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "QJ0001"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EIEI0002"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EI0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FIFI0002"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FI0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EIFI0002"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EI0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FIEI0002"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FI0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EJEJ0002"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EJ0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FJFJ0002"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FJ0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EJFJ0002"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EJ0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FJEJ0002"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FJ0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "PI20002"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "PI0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "QI20002"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "QI0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "PJ20002"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "PJ0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "QJ20002"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "QJ0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EIEI0003"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EI0003"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FIFI0003"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FI0003"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EIFI0003"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EI0003"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FIEI0003"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FI0003"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0002"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EJEJ0003"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EJ0003"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FJFJ0003"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FJ0003"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EJFJ0003"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EJ0003"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FJEJ0003"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FJ0003"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "PI20003"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "PI0003"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "QI20003"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "QI0003"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "PJ20003"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "PJ0003"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "QJ20003"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "QJ0003"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EIEI0004"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EI0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0003"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FIFI0004"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FI0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0003"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EIFI0004"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EI0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0003"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FIEI0004"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FI0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0003"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EJEJ0004"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EJ0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FJFJ0004"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FJ0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EJFJ0004"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EJ0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FJEJ0004"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FJ0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "PI20004"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "PI0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "QI20004"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "QI0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "PJ20004"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "PJ0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "QJ20004"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "QJ0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EIEI0005"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EI0005"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FIFI0005"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FI0005"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EIFI0005"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EI0005"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FIEI0005"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FI0005"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0004"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EJEJ0005"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EJ0005"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0007"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FJFJ0005"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FJ0005"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0007"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EJFJ0005"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EJ0005"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0007"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FJEJ0005"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FJ0005"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0007"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "PI20005"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "PI0005"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "QI20005"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "QI0005"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "PJ20005"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "PJ0005"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "QJ20005"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "QJ0005"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EIEI0006"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EI0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0005"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FIFI0006"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FI0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0005"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EIFI0006"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EI0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0005"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FIEI0006"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FI0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0005"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EJEJ0006"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EJ0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FJFJ0006"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FJ0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EJFJ0006"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EJ0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FJEJ0006"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FJ0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "PI20006"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "PI0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "QI20006"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "QI0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "PJ20006"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "PJ0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "QJ20006"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "QJ0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EIEI0007"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EI0007"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FIFI0007"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FI0007"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EIFI0007"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EI0007"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FIEI0007"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FI0007"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0006"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EJEJ0007"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EJ0007"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0007"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FJFJ0007"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FJ0007"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0007"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "EJFJ0007"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "EJ0007"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VF0007"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "FJEJ0007"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXTIMESY")
        ielftype = arrset(ielftype, ie, iet_["eXTIMESY"])
        vname = "FJ0007"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "VE0007"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "PI20007"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "PI0007"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "QI20007"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "QI0007"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "PJ20007"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "PJ0007"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "QJ20007"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eXSQUARE")
        ielftype = arrset(ielftype, ie, iet_["eXSQUARE"])
        vname = "QJ0007"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        # %%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt = []
        for ig in np.arange(0, ngrp):
            self.grelt.append(np.array([]))
        self.grftype = np.array([])
        self.grelw = []
        nlc = np.array([])
        ig = ig_["GV20001"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["E20001"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["F20001"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        ig = ig_["GV20002"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["E20002"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["F20002"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        ig = ig_["GV20003"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["E20003"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["F20003"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        ig = ig_["GV20004"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["E20004"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["F20004"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        ig = ig_["GV20005"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["E20005"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["F20005"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        ig = ig_["GV20006"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["E20006"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["F20006"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        ig = ig_["GV20007"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["E20007"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["F20007"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        ig = ig_["GPI0001"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EIEI0001"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FIFI0001"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        ig = ig_["GQI0001"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EIFI0001"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FIEI0001"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GPJ0001"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EJEJ0001"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FJFJ0001"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        ig = ig_["GQJ0001"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EJFJ0001"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FJEJ0001"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GMXI0001"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["PI20001"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["QI20001"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GMXJ0001"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["PJ20001"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["QJ20001"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GPI0002"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EIEI0002"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FIFI0002"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        ig = ig_["GQI0002"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EIFI0002"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FIEI0002"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GPJ0002"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EJEJ0002"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FJFJ0002"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        ig = ig_["GQJ0002"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EJFJ0002"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FJEJ0002"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GMXI0002"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["PI20002"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["QI20002"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GMXJ0002"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["PJ20002"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["QJ20002"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GPI0003"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EIEI0003"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FIFI0003"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        ig = ig_["GQI0003"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EIFI0003"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FIEI0003"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GPJ0003"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EJEJ0003"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FJFJ0003"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        ig = ig_["GQJ0003"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EJFJ0003"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FJEJ0003"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GMXI0003"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["PI20003"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["QI20003"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GMXJ0003"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["PJ20003"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["QJ20003"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GPI0004"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EIEI0004"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FIFI0004"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        ig = ig_["GQI0004"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EIFI0004"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FIEI0004"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GPJ0004"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EJEJ0004"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FJFJ0004"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        ig = ig_["GQJ0004"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EJFJ0004"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FJEJ0004"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GMXI0004"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["PI20004"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["QI20004"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GMXJ0004"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["PJ20004"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["QJ20004"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GPI0005"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EIEI0005"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FIFI0005"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        ig = ig_["GQI0005"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EIFI0005"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FIEI0005"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GPJ0005"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EJEJ0005"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FJFJ0005"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        ig = ig_["GQJ0005"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EJFJ0005"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FJEJ0005"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GMXI0005"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["PI20005"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["QI20005"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GMXJ0005"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["PJ20005"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["QJ20005"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GPI0006"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EIEI0006"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FIFI0006"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        ig = ig_["GQI0006"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EIFI0006"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FIEI0006"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GPJ0006"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EJEJ0006"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FJFJ0006"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        ig = ig_["GQJ0006"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EJFJ0006"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FJEJ0006"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GMXI0006"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["PI20006"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["QI20006"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GMXJ0006"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["PJ20006"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["QJ20006"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GPI0007"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EIEI0007"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FIFI0007"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        ig = ig_["GQI0007"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EIFI0007"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FIEI0007"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GPJ0007"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EJEJ0007"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FJFJ0007"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        ig = ig_["GQJ0007"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["EJFJ0007"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["FJEJ0007"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GMXI0007"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["PI20007"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["QI20007"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        ig = ig_["GMXJ0007"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["PJ20007"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["QJ20007"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(1.000))
        # %%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        #    Solutions
        # LO SOLTN               6.78605619D-2
        # LO SOLTN               6.78422730D-2
        # %%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = csr_matrix((valA, (irA, icA)), shape=(ngrp, self.n))
        # %%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        # %%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = np.full((self.m, 1), -float("Inf"))
        self.cupper = np.full((self.m, 1), +float("Inf"))
        self.cupper[np.arange(self.nle)] = np.zeros((self.nle, 1))
        self.clower[np.arange(self.nle, self.nle + self.neq)] = np.zeros((self.neq, 1))
        self.cupper[np.arange(self.nle, self.nle + self.neq)] = np.zeros((self.neq, 1))
        # %%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons = np.where(np.isin(self.congrps, np.setdiff1d(self.congrps, nlc)))[
            0
        ]
        self.pbclass = "C-CLQR2-RN-83-92"
        self.objderlvl = 2
        self.conderlvl = [2]

    # **********************
    #  SET UP THE FUNCTION *
    #  AND RANGE ROUTINES  *
    # **********************

    # %%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eXTIMESY(self, nargout, *args):

        import numpy as np

        EV_ = args[0]
        iel_ = args[1]
        f_ = EV_[0] * EV_[1]
        if not isinstance(f_, float):
            f_ = f_.item()
        if nargout > 1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = np.zeros(dim)
            g_[0] = EV_[1]
            g_[1] = EV_[0]
            if nargout > 2:
                H_ = np.zeros((2, 2))
                H_[0, 1] = 1.0
                H_[1, 0] = H_[0, 1]
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_, g_
        elif nargout == 3:
            return f_, g_, H_

    @staticmethod
    def eXSQUARE(self, nargout, *args):

        import numpy as np

        EV_ = args[0]
        iel_ = args[1]
        f_ = EV_[0] * EV_[0]
        if not isinstance(f_, float):
            f_ = f_.item()
        if nargout > 1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = np.zeros(dim)
            g_[0] = EV_[0] + EV_[0]
            if nargout > 2:
                H_ = np.zeros((1, 1))
                H_[0, 0] = 2.0
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_, g_
        elif nargout == 3:
            return f_, g_, H_


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
