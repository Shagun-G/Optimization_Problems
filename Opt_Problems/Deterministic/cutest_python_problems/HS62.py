from Opt_Problems.Deterministic.s2mpjlib import *


class HS62(CUTEst_problem):

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    #
    #    Problem : HS62
    #    *********
    #
    #    Source: problem 62 in
    #    W. Hock and K. Schittkowski,
    #    "Test examples for nonlinear programming codes",
    #    Lectures Notes in Economics and Mathematical Systems 187, Springer
    #    Verlag, Heidelberg, 1981.
    #
    #    SIF input: J-M Collin and Ph. Toint, April 1990.
    #
    #    classification = "C-COLR2-AY-3-1"
    #
    #    Number of variables
    #
    #
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #   Translated to Python by S2MPJ version 25 XI 2024
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = "HS62"

    def __init__(self, *args):
        import numpy as np
        from scipy.sparse import csr_matrix

        nargin = len(args)

        # %%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_ = {}
        ix_ = {}
        ig_ = {}
        v_["N"] = 3
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
        [ig, ig_, _] = s2mpj_ii("OBJ", ig_)
        gtype = arrset(gtype, ig, "<>")
        [ig, ig_, _] = s2mpj_ii("C1", ig_)
        gtype = arrset(gtype, ig, "==")
        cnames = arrset(cnames, ig, "C1")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["X1"]])
        valA = np.append(valA, float(1.0))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["X2"]])
        valA = np.append(valA, float(1.0))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["X3"]])
        valA = np.append(valA, float(1.0))
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
        self.gconst = arrset(self.gconst, ig_["C1"], float(1.0))
        # %%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = np.zeros((self.n, 1))
        self.xupper = np.full((self.n, 1), float("inf"))
        for I in range(int(v_["1"]), int(v_["N"]) + 1):
            self.xlower[ix_["X" + str(I)]] = 0.0
            self.xupper[ix_["X" + str(I)]] = 1.0
        # %%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = np.zeros((self.n, 1))
        self.y0 = np.zeros((self.m, 1))
        if "X1" in ix_:
            self.x0[ix_["X1"]] = float(0.7)
        else:
            self.y0 = arrset(
                self.y0, findfirst(self.congrps, lambda x: x == ig_["X1"]), float(0.7)
            )
        if "X2" in ix_:
            self.x0[ix_["X2"]] = float(0.2)
        else:
            self.y0 = arrset(
                self.y0, findfirst(self.congrps, lambda x: x == ig_["X2"]), float(0.2)
            )
        if "X3" in ix_:
            self.x0[ix_["X3"]] = float(0.1)
        else:
            self.y0 = arrset(
                self.y0, findfirst(self.congrps, lambda x: x == ig_["X3"]), float(0.1)
            )
        # %%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_ = {}
        elftv = []
        [it, iet_, _] = s2mpj_ii("eLN3N", iet_)
        elftv = loaset(elftv, it, 0, "X")
        elftv = loaset(elftv, it, 1, "Y")
        elftv = loaset(elftv, it, 2, "Z")
        [it, iet_, _] = s2mpj_ii("eLN3D", iet_)
        elftv = loaset(elftv, it, 0, "X")
        elftv = loaset(elftv, it, 1, "Y")
        elftv = loaset(elftv, it, 2, "Z")
        [it, iet_, _] = s2mpj_ii("eLN2N", iet_)
        elftv = loaset(elftv, it, 0, "X")
        elftv = loaset(elftv, it, 1, "Y")
        [it, iet_, _] = s2mpj_ii("eLN2D", iet_)
        elftv = loaset(elftv, it, 0, "X")
        elftv = loaset(elftv, it, 1, "Y")
        [it, iet_, _] = s2mpj_ii("eLN1N", iet_)
        elftv = loaset(elftv, it, 0, "X")
        [it, iet_, _] = s2mpj_ii("eLN1D", iet_)
        elftv = loaset(elftv, it, 0, "X")
        # %%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = np.array([])
        ielftype = np.array([])
        self.elvar = []
        ename = "OE1N"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eLN3N")
        ielftype = arrset(ielftype, ie, iet_["eLN3N"])
        vname = "X1"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "X2"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "X3"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Z")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "OE1D"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eLN3D")
        ielftype = arrset(ielftype, ie, iet_["eLN3D"])
        vname = "X1"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "X2"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "X3"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Z")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "OE2N"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eLN2N")
        ielftype = arrset(ielftype, ie, iet_["eLN2N"])
        vname = "X2"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "X3"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "OE2D"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eLN2D")
        ielftype = arrset(ielftype, ie, iet_["eLN2D"])
        vname = "X2"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "X3"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "Y")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "OE3N"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eLN1N")
        ielftype = arrset(ielftype, ie, iet_["eLN1N"])
        vname = "X3"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
        posev = np.where(elftv[ielftype[ie]] == "X")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        ename = "OE3D"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "eLN1D")
        ielftype = arrset(ielftype, ie, iet_["eLN1D"])
        vname = "X3"
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
        ig = ig_["OBJ"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["OE1N"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-8204.37))
        posel = posel + 1
        self.grelt = loaset(self.grelt, ig, posel, ie_["OE1D"])
        self.grelw = loaset(self.grelw, ig, posel, float(8204.37))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["OE2N"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-9008.72))
        posel = posel + 1
        self.grelt = loaset(self.grelt, ig, posel, ie_["OE2D"])
        self.grelw = loaset(self.grelw, ig, posel, float(9008.72))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["OE3N"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, float(-9330.46))
        posel = posel + 1
        self.grelt = loaset(self.grelt, ig, posel, ie_["OE3D"])
        self.grelw = loaset(self.grelw, ig, posel, float(9330.46))
        # %%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        #    Solution
        # LO SOLTN               -26272.514
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
        self.pbclass = "C-COLR2-AY-3-1"
        self.objderlvl = 2
        self.conderlvl = [2]

    # **********************
    #  SET UP THE FUNCTION *
    #  AND RANGE ROUTINES  *
    # **********************

    # %%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eLN3N(self, nargout, *args):

        import numpy as np

        EV_ = args[0]
        iel_ = args[1]
        U_ = np.zeros((1, 3))
        IV_ = np.zeros(1)
        U_[0, 0] = U_[0, 0] + 1
        U_[0, 1] = U_[0, 1] + 1
        U_[0, 2] = U_[0, 2] + 1
        IV_[0] = U_[0:1, :].dot(EV_)
        NUM = IV_[0] + 0.03
        f_ = np.log(NUM)
        if not isinstance(f_, float):
            f_ = f_.item()
        if nargout > 1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = np.zeros(dim)
            g_[0] = 1.0 / NUM
            g_ = U_.T.dot(g_)
            if nargout > 2:
                H_ = np.zeros((1, 1))
                H_[0, 0] = -1.0 / NUM**2
                H_ = U_.T.dot(H_).dot(U_)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_, g_
        elif nargout == 3:
            return f_, g_, H_

    @staticmethod
    def eLN3D(self, nargout, *args):

        import numpy as np

        EV_ = args[0]
        iel_ = args[1]
        U_ = np.zeros((1, 3))
        IV_ = np.zeros(1)
        U_[0, 0] = U_[0, 0] + 9.000000e-02
        U_[0, 1] = U_[0, 1] + 1
        U_[0, 2] = U_[0, 2] + 1
        IV_[0] = U_[0:1, :].dot(EV_)
        NUM = IV_[0] + 0.03
        f_ = np.log(NUM)
        if not isinstance(f_, float):
            f_ = f_.item()
        if nargout > 1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = np.zeros(dim)
            g_[0] = 1.0 / NUM
            g_ = U_.T.dot(g_)
            if nargout > 2:
                H_ = np.zeros((1, 1))
                H_[0, 0] = -1.0 / NUM**2
                H_ = U_.T.dot(H_).dot(U_)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_, g_
        elif nargout == 3:
            return f_, g_, H_

    @staticmethod
    def eLN2N(self, nargout, *args):

        import numpy as np

        EV_ = args[0]
        iel_ = args[1]
        U_ = np.zeros((1, 2))
        IV_ = np.zeros(1)
        U_[0, 0] = U_[0, 0] + 1
        U_[0, 1] = U_[0, 1] + 1
        IV_[0] = U_[0:1, :].dot(EV_)
        NUM = IV_[0] + 0.03
        f_ = np.log(NUM)
        if not isinstance(f_, float):
            f_ = f_.item()
        if nargout > 1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = np.zeros(dim)
            g_[0] = 1.0 / NUM
            g_ = U_.T.dot(g_)
            if nargout > 2:
                H_ = np.zeros((1, 1))
                H_[0, 0] = -1.0 / NUM**2
                H_ = U_.T.dot(H_).dot(U_)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_, g_
        elif nargout == 3:
            return f_, g_, H_

    @staticmethod
    def eLN2D(self, nargout, *args):

        import numpy as np

        EV_ = args[0]
        iel_ = args[1]
        U_ = np.zeros((1, 2))
        IV_ = np.zeros(1)
        U_[0, 0] = U_[0, 0] + 7.000000e-02
        U_[0, 1] = U_[0, 1] + 1
        IV_[0] = U_[0:1, :].dot(EV_)
        NUM = IV_[0] + 0.03
        f_ = np.log(NUM)
        if not isinstance(f_, float):
            f_ = f_.item()
        if nargout > 1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = np.zeros(dim)
            g_[0] = 1.0 / NUM
            g_ = U_.T.dot(g_)
            if nargout > 2:
                H_ = np.zeros((1, 1))
                H_[0, 0] = -1.0 / NUM**2
                H_ = U_.T.dot(H_).dot(U_)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_, g_
        elif nargout == 3:
            return f_, g_, H_

    @staticmethod
    def eLN1N(self, nargout, *args):

        import numpy as np

        EV_ = args[0]
        iel_ = args[1]
        NUM = EV_[0] + 0.03
        f_ = np.log(NUM)
        if not isinstance(f_, float):
            f_ = f_.item()
        if nargout > 1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = np.zeros(dim)
            g_[0] = 1.0 / NUM
            if nargout > 2:
                H_ = np.zeros((1, 1))
                H_[0, 0] = -1.0 / NUM**2
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_, g_
        elif nargout == 3:
            return f_, g_, H_

    @staticmethod
    def eLN1D(self, nargout, *args):

        import numpy as np

        EV_ = args[0]
        iel_ = args[1]
        NUM = 0.13 * EV_[0] + 0.03
        f_ = np.log(NUM)
        if not isinstance(f_, float):
            f_ = f_.item()
        if nargout > 1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = np.zeros(dim)
            g_[0] = 0.13 / NUM
            if nargout > 2:
                H_ = np.zeros((1, 1))
                H_[0, 0] = -0.0169 / NUM**2
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_, g_
        elif nargout == 3:
            return f_, g_, H_


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
