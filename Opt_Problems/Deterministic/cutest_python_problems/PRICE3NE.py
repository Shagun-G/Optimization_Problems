from Opt_Problems.Deterministic.s2mpjlib import *


class PRICE3NE(CUTEst_problem):

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    #
    #    Problem : PRICE3NE
    #    *********
    #
    #    SCIPY global optimization benchmark example Price03
    #
    #    Fit: y  =  (10(x_1^2-x_2),sqrt(6)(6.4(x_2-0.5)^2 -x_1)) +  e
    #
    #    Source:  Problem from the SCIPY benchmark set
    #      https://github.com/scipy/scipy/tree/master/benchmarks/ ...
    #              benchmarks/go_benchmark_functions
    #
    #    Nonlinear-equation formulation of PRICE3.SIF
    #
    #    SIF input: Nick Gould, Jan 2020
    #
    #    classification = "C-CNOR2-MN-2-2"
    #
    #    Number of data values
    #
    #
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #   Translated to Python by S2MPJ version 25 XI 2024
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = "PRICE3NE"

    def __init__(self, *args):
        import numpy as np
        from scipy.sparse import csr_matrix

        nargin = len(args)

        # %%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_ = {}
        ix_ = {}
        ig_ = {}
        v_["M"] = 2
        v_["N"] = 2
        v_["1"] = 1
        # %%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = np.array([])
        self.xscale = np.array([])
        intvars = np.array([])
        binvars = np.array([])
        irA = np.array([], dtype=int)
        icA = np.array([], dtype=int)
        valA = np.array([], dtype=float)
        for I in range(int(v_["1"]), int(v_["N"]) + 1):
            [iv, ix_, _] = s2mpj_ii("X" + str(I), ix_)
            self.xnames = arrset(self.xnames, iv, "X" + str(I))
        # %%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale = np.array([])
        self.grnames = np.array([])
        cnames = np.array([])
        self.cnames = np.array([])
        gtype = np.array([])
        [ig, ig_, _] = s2mpj_ii("F1", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "F1")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["X2"]])
        valA = np.append(valA, float(-1.0))
        self.gscale = arrset(self.gscale, ig, float(0.1))
        [ig, ig_, _] = s2mpj_ii("F2", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "F2")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["X1"]])
        valA = np.append(valA, float(-1.0))
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
        self.gconst = arrset(self.gconst, ig_["F2"], float(0.6))
        # %%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = np.full((self.n, 1), -float("Inf"))
        self.xupper = np.full((self.n, 1), +float("Inf"))
        # %%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = np.zeros((self.n, 1))
        self.y0 = np.zeros((self.m, 1))
        if "X1" in ix_:
            self.x0[ix_["X1"]] = float(1.0)
        else:
            self.y0 = arrset(
                self.y0, findfirst(self.congrps, lambda x: x == ig_["X1"]), float(1.0)
            )
        if "X2" in ix_:
            self.x0[ix_["X2"]] = float(5.0)
        else:
            self.y0 = arrset(
                self.y0, findfirst(self.congrps, lambda x: x == ig_["X2"]), float(5.0)
            )
        # %%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_ = {}
        elftv = []
        [it, iet_, _] = s2mpj_ii("eSQR", iet_)
        elftv = loaset(elftv, it, 0, "X")
        [it, iet_, _] = s2mpj_ii("eSSQR", iet_)
        elftv = loaset(elftv, it, 0, "X")
        # %%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = np.array([])
        ielftype = np.array([])
        self.elvar = []
        ename = "E1"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eSQR")
        ielftype = arrset(ielftype, ie, iet_["eSQR"])
        vname = "X1"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "E2"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eSSQR")
        ielftype = arrset(ielftype, ie, iet_["eSSQR"])
        vname = "X2"
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
        ig = ig_["F1"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["E1"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, 1.0)
        ig = ig_["F2"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["E2"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(6.4))
        # %%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        #    Least square problems are bounded below by zero
        self.objlower = 0.0
        #    Solution
        # LO SOLUTION            0.0
        # %%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = csr_matrix((valA, (irA, icA)), shape=(ngrp, self.n))
        # %%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        # %%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = np.full((self.m, 1), -float("Inf"))
        self.cupper = np.full((self.m, 1), +float("Inf"))
        self.clower[np.arange(self.nle, self.nle + self.neq)] = np.zeros((self.neq, 1))
        self.cupper[np.arange(self.nle, self.nle + self.neq)] = np.zeros((self.neq, 1))
        # %%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons = np.where(np.isin(self.congrps, np.setdiff1d(self.congrps, nlc)))[
            0
        ]
        self.pbclass = "C-CNOR2-MN-2-2"
        self.objderlvl = 2
        self.conderlvl = [2]

    # **********************
    #  SET UP THE FUNCTION *
    #  AND RANGE ROUTINES  *
    # **********************

    # %%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eSQR(self, nargout, *args):

        import numpy as np

        EV_ = args[0]
        iel_ = args[1]
        f_ = EV_[0] ** 2
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

    @staticmethod
    def eSSQR(self, nargout, *args):

        import numpy as np

        EV_ = args[0]
        iel_ = args[1]
        f_ = (EV_[0] - 0.5) ** 2
        if not isinstance(f_, float):
            f_ = f_.item()
        if nargout > 1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = np.zeros(dim)
            g_[0] = EV_[0] + EV_[0] - 1.0
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
