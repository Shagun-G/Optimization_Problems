from Opt_Problems.Deterministic.s2mpjlib import *


class OSCIPATH(CUTEst_problem):

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    #
    #    Problem : OSCIPATH
    #    *********
    #
    #    An "oscillating path" problem due to Yurii Nesterov
    #
    #    SIF input: Nick Gould, Dec 2006.
    #
    #    classification = "C-CSUR2-AN-V-0"
    #
    #    Number of variables
    #
    #           Alternative values for the SIF file parameters:
    # IE N                   2              $-PARAMETER
    # IE N                   5              $-PARAMETER
    # IE N                   10             $-PARAMETER
    # IE N                   25             $-PARAMETER
    # IE N                   100            $-PARAMETER
    # IE N                   500            $-PARAMETER
    #
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #   Translated to Python by S2MPJ version 25 XI 2024
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = "OSCIPATH"

    def __init__(self, *args):
        import numpy as np
        from scipy.sparse import csr_matrix

        nargin = len(args)

        # %%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_ = {}
        ix_ = {}
        ig_ = {}
        if nargin < 1:
            v_["N"] = int(10)
            #  SIF file default value
        else:
            v_["N"] = int(args[0])
        # RE RHO                 1.0            $-PARAMETER    Nesterov's original value
        if nargin < 2:
            v_["RHO"] = float(500.0)
            #  SIF file default value
        else:
            v_["RHO"] = float(args[1])
        v_["1"] = 1
        v_["2"] = 2
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
        for I in range(int(v_["1"]), int(v_["N"]) + 1):
            [ig, ig_, _] = s2mpj_ii("Q" + str(I), ig_)
            gtype = arrset(gtype, ig, "<>")
            irA = np.append(irA, [ig])
            icA = np.append(icA, [ix_["X" + str(I)]])
            valA = np.append(valA, float(1.0))
        # %%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n = len(ix_)
        ngrp = len(ig_)
        self.objgrps = np.arange(ngrp)
        self.m = 0
        # %%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = np.zeros((ngrp, 1))
        self.gconst = arrset(self.gconst, ig_["Q1"], float(1.0))
        # %%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = np.full((self.n, 1), -float("Inf"))
        self.xupper = np.full((self.n, 1), +float("Inf"))
        # %%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = np.zeros((self.n, 1))
        self.x0[ix_["X" + str(int(v_["1"]))]] = float(-1.0)
        for I in range(int(v_["2"]), int(v_["N"]) + 1):
            self.x0[ix_["X" + str(I)]] = float(1.0)
        # %%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_ = {}
        elftv = []
        [it, iet_, _] = s2mpj_ii("eCHEB", iet_)
        elftv = loaset(elftv, it, 0, "TAU")
        # %%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = np.array([])
        ielftype = np.array([])
        self.elvar = []
        for I in range(int(v_["2"]), int(v_["N"]) + 1):
            v_["I-1"] = -1 + I
            ename = "P" + str(I)
            [ie, ie_, newelt] = s2mpj_ii(ename, ie_)
            if newelt:
                self.elftype = arrset(self.elftype, ie, "eCHEB")
                ielftype = arrset(ielftype, ie, iet_["eCHEB"])
            vname = "X" + str(int(v_["I-1"]))
            [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
            posev = np.where(elftv[ielftype[ie]] == "TAU")[0]
            self.elvar = loaset(self.elvar, ie, posev[0], iv)
        # %%%%%%%%%%%%%%%%%%%%% GRFTYPE %%%%%%%%%%%%%%%%%%%%
        igt_ = {}
        [it, igt_, _] = s2mpj_ii("gPL2", igt_)
        [it, igt_, _] = s2mpj_ii("gPL2", igt_)
        grftp = []
        grftp = loaset(grftp, it, 0, "P")
        # %%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt = []
        for ig in np.arange(0, ngrp):
            self.grelt.append(np.array([]))
        self.grftype = np.array([])
        self.grelw = []
        nlc = np.array([])
        self.grpar = []
        for ig in range(0, ngrp):
            self.grftype = arrset(self.grftype, ig, "gPL2")
        ig = ig_["Q1"]
        posgp = np.where(grftp[igt_[self.grftype[ig]]] == "P")[0]
        self.grpar = loaset(self.grpar, ig, posgp[0], float(0.25))
        for I in range(int(v_["2"]), int(v_["N"]) + 1):
            ig = ig_["Q" + str(I)]
            posel = len(self.grelt[ig])
            self.grelt = loaset(self.grelt, ig, posel, ie_["P" + str(I)])
            self.grelw = loaset(self.grelw, ig, posel, float(-1.0))
            posgp = np.where(grftp[igt_[self.grftype[ig]]] == "P")[0]
            self.grpar = loaset(self.grpar, ig, posgp[0], float(v_["RHO"]))
        # %%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
        #    Solution
        # LO SOLTN                0.0
        # %%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = csr_matrix((valA, (irA, icA)), shape=(ngrp, self.n))
        # %%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        # %%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass = "C-CSUR2-AN-V-0"
        self.objderlvl = 2

    # **********************
    #  SET UP THE FUNCTION *
    #  AND RANGE ROUTINES  *
    # **********************

    # %%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eCHEB(self, nargout, *args):

        import numpy as np

        EV_ = args[0]
        iel_ = args[1]
        f_ = 2.0 * EV_[0] ** 2 - 1.0
        if not isinstance(f_, float):
            f_ = f_.item()
        if nargout > 1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = np.zeros(dim)
            g_[0] = 4.0 * EV_[0]
            if nargout > 2:
                H_ = np.zeros((1, 1))
                H_[0, 0] = 4.0
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_, g_
        elif nargout == 3:
            return f_, g_, H_

    # %%%%%%%%%%%%%%%%% NONLINEAR GROUPS  %%%%%%%%%%%%%%%

    @staticmethod
    def gPL2(self, nargout, *args):

        GVAR_ = args[0]
        igr_ = args[1]
        f_ = self.grpar[igr_][0] * GVAR_ * GVAR_
        if nargout > 1:
            g_ = 2.0 * self.grpar[igr_][0] * GVAR_
            if nargout > 2:
                H_ = np.zeros((1, 1))
                H_ = 2.0 * self.grpar[igr_][0]
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_, g_
        elif nargout == 3:
            return f_, g_, H_


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
