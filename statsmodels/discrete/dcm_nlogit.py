# -*- coding: utf-8 -*-
"""
Nested Logit

Sources: sandbox-statsmodels: treewalkerclass.py

General References
--------------------

Greene, W. `Econometric Analysis`. Prentice Hall, 5th. edition. 2003.
Hess, Florian, 2002, Structural Choice analysis with nested logit models,
    The Stats Journal 2(3) pp 227-252
Train, K. `Discrete Choice Methods with Simulation`.
    Cambridge University Press. 2003

--------------------
"""
import numpy as np
import pandas as pd
from statsmodels.base.model import LikelihoodModel
import time
from collections import OrderedDict

import os
from dcm_clogit import CLogit

# TODO: need some fixes to have minimal functionality
# TODO: clean data handling
# TODO: work on TODO recursive tree structure
# TODO: improve model
# TODO: add TAU warnings for consistent with utility maximization

# note: set values for tau's = 1 on loglike for replicate clogit


def getbranches(tree):
    '''
    walk tree to get list of branches

    Parameters
    ----------
    tree : list of tuples
        tree as defined for RU2NMNL

    Returns
    -------
    branch : list
        list of all branch names

    '''
    if type(tree) == tuple:
        name, subtree = tree
        a = [name]
        for st in subtree:
            a.extend(getbranches(st))
        return a
    return []


def getnodes(tree):
    '''
    walk tree to get list of branches and list of leaves

    Parameters
    ----------
    tree : list of tuples
        tree as defined for RU2NMNL

    Returns
    -------
    branch : list
        list of all branch names
    leaves : list
        list of all leaves names

    '''
    if type(tree) == tuple:
        name, subtree = tree
        ab = [name]
        al = []
        #degenerate branches
        if len(subtree) == 1:
            adeg = [name]
        else:
            adeg = []

        for st in subtree:
            b, l, d = getnodes(st)
            ab.extend(b)
            al.extend(l)
            adeg.extend(d)
        return ab, al, adeg
    return [], [tree], []


class NLogit(LikelihoodModel):
    __doc__ = """
    Random utility maximization nested logit (RUMNL o UMNL)
    RU2 Normalization: moving Scaling Down to the Twig Level
    FIML (Full information ML): Fit the entire model at once, imposing all restrictions
    (fixed (temporaly) normalized at the top (tau top = 1 ))
    # TODO add example to show how values are independent of the normalization

    See treewalkerclass.py
    """

    def __init__(self, endog_data, exog_data, V, ncommon, ref_level,
                         tree, paramsind,
                         name_intercept = None, **kwds):

        self.endog_data = endog_data
        self.exog_data = exog_data

        self.V = V
        self.ncommon = ncommon

        self.ref_level = ref_level

        self.tree = tree
        self.paramsind = paramsind

        self.branchsum = ''
        self.probs = {}
        self.probstxt = {}
        self.branchleaves = {}
        self.branchvalues = {}  # just to keep track of returns by branches
        self.branchsums = {}
        self.bprobs = {}
        self.branches, self.leaves, self.branches_degenerate = getnodes(tree)
        self.nbranches = len(self.branches)

        if name_intercept == None:
            self.exog_data['Intercept'] = 1
            self.name_intercept = 'Intercept'
        else:
            self.name_intercept = name_intercept

        self._initialize()
        super(NLogit, self).__init__(endog = endog_data,
                exog = self.exog_matrix, **kwds)


        #copied over but not quite sure yet
        #unique, parameter array names,
        #sorted alphabetically, order is/should be only internal
        self.paramsnames = (sorted(set([i for j in paramsind.values()
                                       for i in j])) +
                            ['tau_%s' % bname for bname in self.branches])
        self.nparams = len(self.paramsnames)

        #mapping coefficient names to indices to unique/parameter array
        self.paramsidx = dict((name, idx) for (idx, name) in
                              enumerate(self.paramsnames))
        #mapping branch and leaf names to index in parameter array
        self.parinddict = dict((k, [self.paramsidx[j] for j in v])
                               for k, v in self.paramsind.items())

    def _initialize(self):
        """
        Preprocesses the data for Nlogit
        """

        self.J = len(self.V)
        self.nobs = self.endog_data.shape[0] / self.J

        # Endog_bychoices
        self.endog_bychoices = self.endog_data.values.reshape(-1, self.J)
        self.endog_bychoices = self.endog_bychoices[0:number_data]

        endogdict = OrderedDict()
        for ii, key in enumerate(self.V.keys()):
            endogdict[key] = self.endog_bychoices[:,ii]

        if not self.V.keys() == self.leaves:
            self.endog_ordered = np.array([endogdict[leaf] for leaf in self.leaves]).T
        else:
            self.endog_ordered = self.endog_bychoices

        # Exog_bychoices
        exog_bychoices = []
        exog_bychoices_names = []
        choice_index = np.array(self.V.keys() * self.nobs)

        for key in iter(self.V):
            (exog_bychoices.append(self.exog_data[self.V[key]]
                                    [choice_index == key].values))

        for key in self.V:
            exog_bychoices_names.append(self.V[key])

        self.exog_bychoices = exog_bychoices

        #TODO
        datadict = {}
        datadict = dict(zip(self.V.keys(),
                        [exog_bychoices[i][0:number_data] for i in range(4)]))

        #for testing only (mock that returns it's own name
        if testxb == 1:
            datadict = dict(zip(['air', 'train', 'bus', 'car'],
                            ['Airdata', 'Traindata', 'Busdata', 'Cardata']))

        self.datadict = datadict

        self.datadict.update({'top' :    [],
                              'fly' :    [],
                              'ground':  []})

        # Betas
        beta_not_common = ([len(exog_bychoices_names[ii]) - self.ncommon
                            for ii in range(self.J)])
        exog_names_prueba = []

        for ii, key in enumerate(self.V):
            exog_names_prueba.append(key * beta_not_common[ii])

        zi = np.r_[[self.ncommon], self.ncommon + np.array(beta_not_common)\
                    .cumsum()]
        z = np.arange(max(zi))
        beta_ind = [np.r_[np.arange(self.ncommon), z[zi[ii]:zi[ii + 1]]]
                               for ii in range(len(zi) - 1)]  # index of betas
        self.beta_ind = beta_ind

        beta_ind_str = ([map(str, beta_ind[ii]) for ii in range(self.J)])
        beta_ind_J = ([map(str, beta_ind[ii]) for ii in range(self.J)])

        for ii in range(self.J):
            for jj, item in enumerate(beta_ind[ii]):
                if item in np.arange(self.ncommon):
                    beta_ind_J[ii][jj] = ''
                else:
                    beta_ind_J[ii][jj] = ' (' + self.V.keys()[ii] + ')'

        self.betas = OrderedDict()

        for sublist in range(self.J):
            aa = []
            for ii in range(len(exog_bychoices_names[sublist])):
                aa.append(
                beta_ind_str[sublist][ii] + ' ' +
                exog_bychoices_names[sublist][ii]
                + beta_ind_J[sublist][ii])
            self.betas[sublist] = aa

        # Exog
        pieces = []
        for ii in range(self.J):
            pieces.append(pd.DataFrame(exog_bychoices[ii],
                                       columns=self.betas[ii]))

        self.exog_matrix_all = (pd.concat(pieces, axis = 0,
                                          keys = self.V.keys(),
                                          names = ['choice', 'nobs'])
                           .fillna(value = 0).sortlevel(1).reset_index())

        self.exog_matrix = self.exog_matrix_all.iloc[:, 2:]

        self.K = len(self.exog_matrix.columns)

        self.df_model = self.K
        self.df_resid = int(self.nobs - self.K)

    def get_probs(self, params):
        '''
        obtain the probability array given an array of parameters

        This is the function that can be called by loglike or other methods
        that need the probabilities as function of the params.

        Parameters
        ----------
        params : 1d array, (nparams,)
            coefficients and tau that parameterize the model. The required
            length can be obtained by nparams. (and will depend on the number
            of degenerate leaves - not yet)

        Returns
        -------
        probs : array, (nobs, nchoices)
            probabilites for all choices for each observation. The order
            is available by attribute leaves. See note in docstring of class



        '''
        if testxb < 2:
            print "fix testxb = 1"
        else:
            self.recursionparams = params
            self.calc_prob(self.tree)
            self.probs_array = np.array([self.probs[leaf] for leaf in self.leaves]).T
            return self.probs_array
            #TODO: need a check/assert that this sequence is the same as the
            #      encoding in endog

    def calc_prob(self, tree, parent=None):
        '''walking a tree bottom-up based on dictionary
        '''

        #0.5#2 #placeholder for now
        #should be tau=self.taus[name] but as part of params for optimization
        datadict = self.datadict
        paramsind = self.paramsind
        branchsum = self.branchsum

        if type(tree) == tuple:   # assumes leaves are int for choice index

            name, subtree = tree
            self.branchleaves[name] = []  # register branch in dictionary

            tau = self.recursionparams[self.paramsidx['tau_' + name]]
            if DEBUG:
                print '----------- starting next branch-----------'
                print name, datadict[name], 'tau=', tau
                print 'subtree', subtree
            branchvalue = []

            if testxb == 2:
                if DEBUG:
                    print "branchsum = 0"
                branchsum = 0

            elif testxb == 1:
                branchsum = datadict[name]

            for b in subtree:
                if DEBUG:
                    print "___starting with (b):", b, ",in branch", name
                bv = self.calc_prob(b, name)
                if DEBUG:
                    print "intial branchsum is", branchsum
                    print "__bv is", bv

                branchvalue.append(bv)
                branchsum = branchsum + bv

                if DEBUG:
                    print "after bv, branchsum is", branchsum

            self.branchvalues[name] = branchvalue   # keep track what was returned

            if DEBUG:
                print '----------- returning to branch-----------',
                print name
                print 'branchsum in branch', name, branchsum

            if parent:
                if DEBUG:
                    print 'parent', parent
                    print '__calc the prob of leaves of', name
                self.branchleaves[parent].extend(self.branchleaves[name])

            if testxb < 2:
                return branchsum
            else:
                self.bprobs[name] = []
                for bidx, b in enumerate(subtree):
                    if DEBUG:
                        print 'bottom branch: repr(b)', repr(b), bidx
                    #if len(b) == 1: #TODO: skip leaves, check this
                    if not type(b) == tuple:  # isinstance(b, str):
                        #TODO: replace this with a check for branch (tuple) instead
                        #this implies name is a bottom branch,
                        #possible to add special things here
                        #TODO: need tau possibly here
                        if DEBUG:
                            print "num-prob", self.probs[b]
                        self.probs[b] = self.probs[b] / branchsum  # normalized prob
                        if DEBUG:
                            print "leaf prob (normalized):", self.probs[b]
                            print '*********** branchsum at bottom branch', branchsum

                    else:
                        bname = b[0]
                        if DEBUG:
                            print '__calc the prob of branches of', name
                        branchsum2 = sum(self.branchvalues[name])
#                        assert np.abs(branchsum - branchsum2).sum() < 1e-8
                        bprob = branchvalue[bidx] / branchsum2  # normalized prob
                        self.bprobs[name].append(bprob)

                        if DEBUG:
                            print "branch:", bname, "num:", bidx
                            print "branchvalue:", branchvalue[bidx], "prob", bprob
                            print '__calc the prob of alternative inside', bname

                        for k in self.branchleaves[bname]:

                            if DEBUG:
                                print k,": leaf prob", self.probs[k], "*",
                                print 'branchprob', bname,  bprob
                                print "    --_> prob alternative", self.probs[k] * bprob
                            self.probs[k] = self.probs[k] * bprob

            if DEBUG:
                print '__now working on @branch@', tree
            if testxb < 2:
                return branchsum
            else:   #this is the relevant part
                self.branchsums[name] = branchsum
                if np.size(self.datadict[name]) > 0:
                    branchxb = np.dot(self.datadict[name],
                                  self.recursionparams[self.parinddict[name]])
                else:
                    branchxb = 0
                if not name == 'top':
                    tau = self.recursionparams[self.paramsidx['tau_' + name]]

                else:
                    tau = 1
#                iv = branchxb + tau * branchsum #which tau: name or parent???
                if DEBUG:
                    print "brach prob:"
                    print  np.exp(branchxb + tau * np.log(branchsum))
                    print "for:", name, "tau:", tau, "brancxb", branchxb

                return  np.exp(branchxb + tau * np.log(branchsum))  # iv
                #branchsum is now IV, TODO: add effect of branch variables

        else:
            if DEBUG:
                print "working on @leaf@", tree
            tau = self.recursionparams[self.paramsidx['tau_' + parent]]

            self.branchleaves[parent].append(tree)   # register leave with parent
            self.probstxt[tree] = [tree + '-prob' +
                                '(%s)' % ', '.join(self.paramsind[tree])]

            if testxb == 2:
                if DEBUG:
                    print 'parent', parent
                    print "for", tree, self.probstxt[tree]
                    #this is not yet a prob, not normalized to 1, it is exp(x*b)
                leafprob = np.exp(np.dot(self.datadict[tree],
                                  self.recursionparams[self.parinddict[tree]])
                              / tau)   # fake tau for now, wrong spot ???
                # it seems I get the same answer with and without tau here
                self.probs[tree] = leafprob  #= 1 #try initialization only
                # TODO: where  should I add tau in the leaves

                if DEBUG:
                    print "numerator leaf prob:", leafprob
                    print "for:", tree, "tau:", tau
                return leafprob
            elif testxb == 1:
                leavessum = np.array(datadict[tree])  # sum((datadict[bi] for bi in datadict[tree]))
                if DEBUG:
                    print 'final branch with', tree, ''.join(tree), leavessum  # sum(tree)
                return leavessum  # sum(xb[tree])
            elif testxb == 0:
                return ''.join(tree)  # sum(tree)

    def loglike(self, params):

        """
        Log-likelihood of the nested logit model.

        Parameters
        ----------
        params : array
            the parameters of the conditional logit model.

        Returns
        -------
        loglike : float
            the log-likelihood function of the model evaluated at `params`.

        Notes
        ------
        .. math:: $$\\ln L=\\sum_{i=1}^{n}\\ln\\left(P_{ij|b}P_{b}\\right)$$

        where :

        """
        params[-self.nbranches:] = 1  # values for tau's clogit
        params[-self.nbranches] = 1  # normalized at the top

        prob = self.get_probs(params)

        loglike = (self.endog_ordered * np.log(prob)).sum(1)
        return loglike.sum()

    def score(self, params):
        """
        """
        from statsmodels.tools.numdiff import approx_fprime
        return approx_fprime(params, self.loglike, epsilon=1e-8)

    def hessian(self, params):
        """
        """
        # TODO
        from statsmodels.tools.numdiff import approx_hess
        return approx_hess(params, self.loglike)

    def fit(self, start_params=None, maxiter=None, maxfun=None,
            method=None, full_output=True, disp=None, callback=None, **kwds):

        """
        Returns
        -------
        Fit object for likelihood based models
        See: GenericLikelihoodModelResults

        """

        if start_params is None:
        # TODO
            CLogit_res = CLogit(self.endog_data, self.exog_data,
                                    self.V, self.ncommon, self.ref_level
                                  ).fit(disp=0)
            idx = np.array([2, 5, 4, 0, 3, 1])

            start_params = np.r_[CLogit_res.params[idx],
                                  np.array([1, 1, 1])]

#           start_params = np.zeros(self.nparams)
#           start_params[-self.nbranches:] = 1  # values for tau's

        else:
            start_params = np.asarray(start_params)

        start_time = time.time()
        model_fit = super(NLogit, self).fit(disp = disp, full_output = full_output,
                                            start_params = start_params,
                                            method=method, maxiter=maxiter,
                                            maxfun=maxfun, **kwds)

        self.params = model_fit.params
        end_time = time.time()
        self.elapsed_time = end_time - start_time
        return model_fit


if __name__ == "__main__":

    print 'Example:'

    # Load data
    from patsy import dmatrices

    url = "http://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/ModeChoice.csv"
    file_ = "ModeChoice.csv"

    if not os.path.exists(file_):
        import urllib
        urllib.urlretrieve(url, "ModeChoice.csv")
    df = pd.read_csv(file_)
    df.describe()

    f = 'mode  ~ ttme+invc+invt+gc+hinc+psize'
    y, X = dmatrices(f, df, return_type='dataframe')

    # Set up model

    # Names of the variables for the utility function for each alternative
    # variables with common coefficients have to be first in each array
    V = OrderedDict((
        ('air',   ['gc', 'ttme', 'Intercept', 'hinc']),
        ('train', ['gc', 'ttme', 'Intercept']),
        ('bus',   ['gc', 'ttme', 'Intercept']),
        ('car',   ['gc', 'ttme']))
        )
    # Number of common coefficients
    ncommon = 2

    # Tree
    tree = ('top',
            [('fly', ['air']),
             ('ground', ['train', 'car', 'bus'])
            ]
        )

    # TODO: set paramsind inside class
    print getbranches(tree)
    print getnodes(tree)

    paramsind = {'top' :   [],
                 'fly' :   [],
                 'ground': [],
                 'air' :   ['gc', 'ttme', 'constA','hinc' ],
                 'train' : ['gc', 'ttme', 'constT'],
                 'bus' :   ['gc', 'ttme', 'constB'],
                 'car' :   ['gc', 'ttme']
                 }

    # TODO: testxb = 1 fix me shapes (0) (210,4)
    testxb = 2  # global to class to return strings(testxb = 1)instead of numbers
    DEBUG = 0
    number_data = 210  # number of data to use [1 -210]
    # Describe model
    nlogit_mod = NLogit(endog_data = y, exog_data = X, V = V,
                        ncommon = ncommon,
                        tree = tree, paramsind = paramsind,
                        ref_level = 'car',
                        name_intercept = 'Intercept')

    # Fit model
    nlogit_res = nlogit_mod.fit( maxiter = 100,
                                 maxfun= 100,
                                 method = "bfgs",
                                 gtol = 10e-8,
                                 retal = True,
                                 disp = True,)

    print nlogit_res.params
    print nlogit_res.llf
