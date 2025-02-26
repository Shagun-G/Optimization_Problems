from Opt_Problems.Deterministic.s2mpjlib import *


class BENNETT5(CUTEst_problem):

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    #
    #    Problem : BENNETT5
    #    *********
    #
    #    NIST Data fitting problem BENNETT5 given as an inconsistent set of
    #    nonlinear equations.
    #
    #    Fit: y = b1 * (b2+x)**(-1/b3) + e
    #
    #    Source:  Problem from the NIST nonlinear regression test set
    #      http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml
    #
    #    Reference:	Bennett, L., L. Swartzendruber, H. Brown, NIST (1994).
    #      Superconductivity Magnetization Modeling.
    #
    #    SIF input: Nick Gould and Tyrone Rees, Oct 2015
    #
    #    classification = "C-CNOR2-MN-3-154"
    #
    #    Number of data values
    #
    #
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #   Translated to Python by S2MPJ version 25 XI 2024
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = "BENNETT5"

    def __init__(self, *args):
        import numpy as np
        from scipy.sparse import csr_matrix

        nargin = len(args)

        # %%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_ = {}
        ix_ = {}
        ig_ = {}
        v_["M"] = 154
        v_["N"] = 3
        v_["1"] = 1
        v_["X1"] = 7.447168e0
        v_["X2"] = 8.102586e0
        v_["X3"] = 8.452547e0
        v_["X4"] = 8.711278e0
        v_["X5"] = 8.916774e0
        v_["X6"] = 9.087155e0
        v_["X7"] = 9.232590e0
        v_["X8"] = 9.359535e0
        v_["X9"] = 9.472166e0
        v_["X10"] = 9.573384e0
        v_["X11"] = 9.665293e0
        v_["X12"] = 9.749461e0
        v_["X13"] = 9.827092e0
        v_["X14"] = 9.899128e0
        v_["X15"] = 9.966321e0
        v_["X16"] = 10.029280e0
        v_["X17"] = 10.088510e0
        v_["X18"] = 10.144430e0
        v_["X19"] = 10.197380e0
        v_["X20"] = 10.247670e0
        v_["X21"] = 10.295560e0
        v_["X22"] = 10.341250e0
        v_["X23"] = 10.384950e0
        v_["X24"] = 10.426820e0
        v_["X25"] = 10.467000e0
        v_["X26"] = 10.505640e0
        v_["X27"] = 10.542830e0
        v_["X28"] = 10.578690e0
        v_["X29"] = 10.613310e0
        v_["X30"] = 10.646780e0
        v_["X31"] = 10.679150e0
        v_["X32"] = 10.710520e0
        v_["X33"] = 10.740920e0
        v_["X34"] = 10.770440e0
        v_["X35"] = 10.799100e0
        v_["X36"] = 10.826970e0
        v_["X37"] = 10.854080e0
        v_["X38"] = 10.880470e0
        v_["X39"] = 10.906190e0
        v_["X40"] = 10.931260e0
        v_["X41"] = 10.955720e0
        v_["X42"] = 10.979590e0
        v_["X43"] = 11.002910e0
        v_["X44"] = 11.025700e0
        v_["X45"] = 11.047980e0
        v_["X46"] = 11.069770e0
        v_["X47"] = 11.091100e0
        v_["X48"] = 11.111980e0
        v_["X49"] = 11.132440e0
        v_["X50"] = 11.152480e0
        v_["X51"] = 11.172130e0
        v_["X52"] = 11.191410e0
        v_["X53"] = 11.210310e0
        v_["X54"] = 11.228870e0
        v_["X55"] = 11.247090e0
        v_["X56"] = 11.264980e0
        v_["X57"] = 11.282560e0
        v_["X58"] = 11.299840e0
        v_["X59"] = 11.316820e0
        v_["X60"] = 11.333520e0
        v_["X61"] = 11.349940e0
        v_["X62"] = 11.366100e0
        v_["X63"] = 11.382000e0
        v_["X64"] = 11.397660e0
        v_["X65"] = 11.413070e0
        v_["X66"] = 11.428240e0
        v_["X67"] = 11.443200e0
        v_["X68"] = 11.457930e0
        v_["X69"] = 11.472440e0
        v_["X70"] = 11.486750e0
        v_["X71"] = 11.500860e0
        v_["X72"] = 11.514770e0
        v_["X73"] = 11.528490e0
        v_["X74"] = 11.542020e0
        v_["X75"] = 11.555380e0
        v_["X76"] = 11.568550e0
        v_["X77"] = 11.581560e0
        v_["X78"] = 11.594420e0
        v_["X79"] = 11.607121e0
        v_["X80"] = 11.619640e0
        v_["X81"] = 11.632000e0
        v_["X82"] = 11.644210e0
        v_["X83"] = 11.656280e0
        v_["X84"] = 11.668200e0
        v_["X85"] = 11.679980e0
        v_["X86"] = 11.691620e0
        v_["X87"] = 11.703130e0
        v_["X88"] = 11.714510e0
        v_["X89"] = 11.725760e0
        v_["X90"] = 11.736880e0
        v_["X91"] = 11.747890e0
        v_["X92"] = 11.758780e0
        v_["X93"] = 11.769550e0
        v_["X94"] = 11.780200e0
        v_["X95"] = 11.790730e0
        v_["X96"] = 11.801160e0
        v_["X97"] = 11.811480e0
        v_["X98"] = 11.821700e0
        v_["X99"] = 11.831810e0
        v_["X100"] = 11.841820e0
        v_["X101"] = 11.851730e0
        v_["X102"] = 11.861550e0
        v_["X103"] = 11.871270e0
        v_["X104"] = 11.880890e0
        v_["X105"] = 11.890420e0
        v_["X106"] = 11.899870e0
        v_["X107"] = 11.909220e0
        v_["X108"] = 11.918490e0
        v_["X109"] = 11.927680e0
        v_["X110"] = 11.936780e0
        v_["X111"] = 11.945790e0
        v_["X112"] = 11.954730e0
        v_["X113"] = 11.963590e0
        v_["X114"] = 11.972370e0
        v_["X115"] = 11.981070e0
        v_["X116"] = 11.989700e0
        v_["X117"] = 11.998260e0
        v_["X118"] = 12.006740e0
        v_["X119"] = 12.015150e0
        v_["X120"] = 12.023490e0
        v_["X121"] = 12.031760e0
        v_["X122"] = 12.039970e0
        v_["X123"] = 12.048100e0
        v_["X124"] = 12.056170e0
        v_["X125"] = 12.064180e0
        v_["X126"] = 12.072120e0
        v_["X127"] = 12.080010e0
        v_["X128"] = 12.087820e0
        v_["X129"] = 12.095580e0
        v_["X130"] = 12.103280e0
        v_["X131"] = 12.110920e0
        v_["X132"] = 12.118500e0
        v_["X133"] = 12.126030e0
        v_["X134"] = 12.133500e0
        v_["X135"] = 12.140910e0
        v_["X136"] = 12.148270e0
        v_["X137"] = 12.155570e0
        v_["X138"] = 12.162830e0
        v_["X139"] = 12.170030e0
        v_["X140"] = 12.177170e0
        v_["X141"] = 12.184270e0
        v_["X142"] = 12.191320e0
        v_["X143"] = 12.198320e0
        v_["X144"] = 12.205270e0
        v_["X145"] = 12.212170e0
        v_["X146"] = 12.219030e0
        v_["X147"] = 12.225840e0
        v_["X148"] = 12.232600e0
        v_["X149"] = 12.239320e0
        v_["X150"] = 12.245990e0
        v_["X151"] = 12.252620e0
        v_["X152"] = 12.259200e0
        v_["X153"] = 12.265750e0
        v_["X154"] = 12.272240e0
        v_["Y1"] = -34.834702e0
        v_["Y2"] = -34.393200e0
        v_["Y3"] = -34.152901e0
        v_["Y4"] = -33.979099e0
        v_["Y5"] = -33.845901e0
        v_["Y6"] = -33.732899e0
        v_["Y7"] = -33.640301e0
        v_["Y8"] = -33.559200e0
        v_["Y9"] = -33.486801e0
        v_["Y10"] = -33.423100e0
        v_["Y11"] = -33.365101e0
        v_["Y12"] = -33.313000e0
        v_["Y13"] = -33.260899e0
        v_["Y14"] = -33.217400e0
        v_["Y15"] = -33.176899e0
        v_["Y16"] = -33.139198e0
        v_["Y17"] = -33.101601e0
        v_["Y18"] = -33.066799e0
        v_["Y19"] = -33.035000e0
        v_["Y20"] = -33.003101e0
        v_["Y21"] = -32.971298e0
        v_["Y22"] = -32.942299e0
        v_["Y23"] = -32.916302e0
        v_["Y24"] = -32.890202e0
        v_["Y25"] = -32.864101e0
        v_["Y26"] = -32.841000e0
        v_["Y27"] = -32.817799e0
        v_["Y28"] = -32.797501e0
        v_["Y29"] = -32.774300e0
        v_["Y30"] = -32.757000e0
        v_["Y31"] = -32.733799e0
        v_["Y32"] = -32.716400e0
        v_["Y33"] = -32.699100e0
        v_["Y34"] = -32.678799e0
        v_["Y35"] = -32.661400e0
        v_["Y36"] = -32.644001e0
        v_["Y37"] = -32.626701e0
        v_["Y38"] = -32.612202e0
        v_["Y39"] = -32.597698e0
        v_["Y40"] = -32.583199e0
        v_["Y41"] = -32.568699e0
        v_["Y42"] = -32.554298e0
        v_["Y43"] = -32.539799e0
        v_["Y44"] = -32.525299e0
        v_["Y45"] = -32.510799e0
        v_["Y46"] = -32.499199e0
        v_["Y47"] = -32.487598e0
        v_["Y48"] = -32.473202e0
        v_["Y49"] = -32.461601e0
        v_["Y50"] = -32.435501e0
        v_["Y51"] = -32.435501e0
        v_["Y52"] = -32.426800e0
        v_["Y53"] = -32.412300e0
        v_["Y54"] = -32.400799e0
        v_["Y55"] = -32.392101e0
        v_["Y56"] = -32.380501e0
        v_["Y57"] = -32.366001e0
        v_["Y58"] = -32.357300e0
        v_["Y59"] = -32.348598e0
        v_["Y60"] = -32.339901e0
        v_["Y61"] = -32.328400e0
        v_["Y62"] = -32.319698e0
        v_["Y63"] = -32.311001e0
        v_["Y64"] = -32.299400e0
        v_["Y65"] = -32.290699e0
        v_["Y66"] = -32.282001e0
        v_["Y67"] = -32.273300e0
        v_["Y68"] = -32.264599e0
        v_["Y69"] = -32.256001e0
        v_["Y70"] = -32.247299e0
        v_["Y71"] = -32.238602e0
        v_["Y72"] = -32.229900e0
        v_["Y73"] = -32.224098e0
        v_["Y74"] = -32.215401e0
        v_["Y75"] = -32.203800e0
        v_["Y76"] = -32.198002e0
        v_["Y77"] = -32.189400e0
        v_["Y78"] = -32.183601e0
        v_["Y79"] = -32.174900e0
        v_["Y80"] = -32.169102e0
        v_["Y81"] = -32.163300e0
        v_["Y82"] = -32.154598e0
        v_["Y83"] = -32.145901e0
        v_["Y84"] = -32.140099e0
        v_["Y85"] = -32.131401e0
        v_["Y86"] = -32.125599e0
        v_["Y87"] = -32.119801e0
        v_["Y88"] = -32.111198e0
        v_["Y89"] = -32.105400e0
        v_["Y90"] = -32.096699e0
        v_["Y91"] = -32.090900e0
        v_["Y92"] = -32.088001e0
        v_["Y93"] = -32.079300e0
        v_["Y94"] = -32.073502e0
        v_["Y95"] = -32.067699e0
        v_["Y96"] = -32.061901e0
        v_["Y97"] = -32.056099e0
        v_["Y98"] = -32.050301e0
        v_["Y99"] = -32.044498e0
        v_["Y100"] = -32.038799e0
        v_["Y101"] = -32.033001e0
        v_["Y102"] = -32.027199e0
        v_["Y103"] = -32.024300e0
        v_["Y104"] = -32.018501e0
        v_["Y105"] = -32.012699e0
        v_["Y106"] = -32.004002e0
        v_["Y107"] = -32.001099e0
        v_["Y108"] = -31.995300e0
        v_["Y109"] = -31.989500e0
        v_["Y110"] = -31.983700e0
        v_["Y111"] = -31.977900e0
        v_["Y112"] = -31.972099e0
        v_["Y113"] = -31.969299e0
        v_["Y114"] = -31.963501e0
        v_["Y115"] = -31.957701e0
        v_["Y116"] = -31.951900e0
        v_["Y117"] = -31.946100e0
        v_["Y118"] = -31.940300e0
        v_["Y119"] = -31.937401e0
        v_["Y120"] = -31.931601e0
        v_["Y121"] = -31.925800e0
        v_["Y122"] = -31.922899e0
        v_["Y123"] = -31.917101e0
        v_["Y124"] = -31.911301e0
        v_["Y125"] = -31.908400e0
        v_["Y126"] = -31.902599e0
        v_["Y127"] = -31.896900e0
        v_["Y128"] = -31.893999e0
        v_["Y129"] = -31.888201e0
        v_["Y130"] = -31.885300e0
        v_["Y131"] = -31.882401e0
        v_["Y132"] = -31.876600e0
        v_["Y133"] = -31.873699e0
        v_["Y134"] = -31.867901e0
        v_["Y135"] = -31.862101e0
        v_["Y136"] = -31.859200e0
        v_["Y137"] = -31.856300e0
        v_["Y138"] = -31.850500e0
        v_["Y139"] = -31.844700e0
        v_["Y140"] = -31.841801e0
        v_["Y141"] = -31.838900e0
        v_["Y142"] = -31.833099e0
        v_["Y143"] = -31.830200e0
        v_["Y144"] = -31.827299e0
        v_["Y145"] = -31.821600e0
        v_["Y146"] = -31.818701e0
        v_["Y147"] = -31.812901e0
        v_["Y148"] = -31.809999e0
        v_["Y149"] = -31.807100e0
        v_["Y150"] = -31.801300e0
        v_["Y151"] = -31.798401e0
        v_["Y152"] = -31.795500e0
        v_["Y153"] = -31.789700e0
        v_["Y154"] = -31.786800e0
        # %%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = np.array([])
        self.xscale = np.array([])
        intvars = np.array([])
        binvars = np.array([])
        irA = np.array([], dtype=int)
        icA = np.array([], dtype=int)
        valA = np.array([], dtype=float)
        for I in range(int(v_["1"]), int(v_["N"]) + 1):
            [iv, ix_, _] = s2mpj_ii("B" + str(I), ix_)
            self.xnames = arrset(self.xnames, iv, "B" + str(I))
        # %%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale = np.array([])
        self.grnames = np.array([])
        cnames = np.array([])
        self.cnames = np.array([])
        gtype = np.array([])
        for I in range(int(v_["1"]), int(v_["M"]) + 1):
            [ig, ig_, _] = s2mpj_ii("F" + str(I), ig_)
            gtype = arrset(gtype, ig, "==")
            cnames = arrset(cnames, ig, "F" + str(I))
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
        for I in range(int(v_["1"]), int(v_["M"]) + 1):
            self.gconst = arrset(
                self.gconst, ig_["F" + str(I)], float(v_["Y" + str(I)])
            )
        # %%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = np.full((self.n, 1), -float("Inf"))
        self.xupper = np.full((self.n, 1), +float("Inf"))
        # %%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = np.zeros((self.n, 1))
        self.y0 = np.zeros((self.m, 1))
        if "B1" in ix_:
            self.x0[ix_["B1"]] = float(-2000.0)
        else:
            self.y0 = arrset(
                self.y0,
                findfirst(self.congrps, lambda x: x == ig_["B1"]),
                float(-2000.0),
            )
        if "B2" in ix_:
            self.x0[ix_["B2"]] = float(50.0)
        else:
            self.y0 = arrset(
                self.y0, findfirst(self.congrps, lambda x: x == ig_["B2"]), float(50.0)
            )
        if "B3" in ix_:
            self.x0[ix_["B3"]] = float(0.8)
        else:
            self.y0 = arrset(
                self.y0, findfirst(self.congrps, lambda x: x == ig_["B3"]), float(0.8)
            )
        pass
        # %%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_ = {}
        elftv = []
        [it, iet_, _] = s2mpj_ii("eE15", iet_)
        elftv = loaset(elftv, it, 0, "V1")
        elftv = loaset(elftv, it, 1, "V2")
        elftv = loaset(elftv, it, 2, "V3")
        elftp = []
        elftp = loaset(elftp, it, 0, "X")
        # %%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = np.array([])
        ielftype = np.array([])
        self.elvar = []
        self.elpar = []
        for I in range(int(v_["1"]), int(v_["M"]) + 1):
            ename = "E" + str(I)
            [ie, ie_, _] = s2mpj_ii(ename, ie_)
            self.elftype = arrset(self.elftype, ie, "eE15")
            ielftype = arrset(ielftype, ie, iet_["eE15"])
            vname = "B1"
            [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
            posev = np.where(elftv[ielftype[ie]] == "V1")[0]
            self.elvar = loaset(self.elvar, ie, posev[0], iv)
            vname = "B2"
            [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
            posev = np.where(elftv[ielftype[ie]] == "V2")[0]
            self.elvar = loaset(self.elvar, ie, posev[0], iv)
            vname = "B3"
            [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, None)
            posev = np.where(elftv[ielftype[ie]] == "V3")[0]
            self.elvar = loaset(self.elvar, ie, posev[0], iv)
            posep = np.where(elftp[ielftype[ie]] == "X")[0]
            self.elpar = loaset(self.elpar, ie, posep[0], float(v_["X" + str(I)]))
        # %%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt = []
        for ig in np.arange(0, ngrp):
            self.grelt.append(np.array([]))
        self.grftype = np.array([])
        self.grelw = []
        nlc = np.array([])
        for I in range(int(v_["1"]), int(v_["M"]) + 1):
            ig = ig_["F" + str(I)]
            posel = len(self.grelt[ig])
            self.grelt = loaset(self.grelt, ig, posel, ie_["E" + str(I)])
            nlc = np.union1d(nlc, np.array([ig]))
            self.grelw = loaset(self.grelw, ig, posel, 1.0)
        # %%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        #    Least square problems are bounded below by zero
        self.objlower = 0.0
        #    Solution
        # LO SOLTN
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
        self.pbclass = "C-CNOR2-MN-3-154"
        self.objderlvl = 2
        self.conderlvl = [2]

    # **********************
    #  SET UP THE FUNCTION *
    #  AND RANGE ROUTINES  *
    # **********************

    # %%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eE15(self, nargout, *args):

        import numpy as np

        EV_ = args[0]
        iel_ = args[1]
        V3INV = 1.0 / EV_[2]
        V2PX = EV_[1] + self.elpar[iel_][0]
        V2PXL = np.log(V2PX)
        V2PXP = V2PX**V3INV
        V2PXP1 = V2PX ** (V3INV + 1.0)
        V2PXP2 = V2PX ** (V3INV + 2.0)
        f_ = EV_[0] / V2PXP
        if not isinstance(f_, float):
            f_ = f_.item()
        if nargout > 1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = np.zeros(dim)
            g_[0] = 1.0 / V2PXP
            g_[1] = -EV_[0] / (EV_[2] * V2PXP1)
            g_[2] = EV_[0] * V2PXL / (V2PXP * EV_[2] ** 2)
            if nargout > 2:
                H_ = np.zeros((3, 3))
                H_[0, 1] = -1.0 / (EV_[2] * V2PXP1)
                H_[1, 0] = H_[0, 1]
                H_[0, 2] = V2PXL / (V2PXP * EV_[2] ** 2)
                H_[2, 0] = H_[0, 2]
                H_[1, 1] = EV_[0] * (1.0 / EV_[2] + 1.0) / (EV_[2] * V2PXP2)
                H_[1, 2] = EV_[0] / (V2PX * V2PXP * EV_[2] ** 2) - EV_[0] * V2PXL / (
                    V2PXP1 * EV_[2] ** 3
                )
                H_[2, 1] = H_[1, 2]
                H_[2, 2] = EV_[0] * V2PXL**2 / (V2PXP * EV_[2] ** 4) - 2.0 * EV_[
                    0
                ] * V2PXL / (V2PXP * EV_[2] ** 3)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_, g_
        elif nargout == 3:
            return f_, g_, H_


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
