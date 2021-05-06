"""Lineage and clustering analysis.

This module provides various solvers to infer lineage and SNV clustering
of tumour samples.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import abc
import argparse
import collections
import copy
import os
import os.path
import pkg_resources
import sys
import traceback

import logbook
import numpy
import pandas
import scipy.misc
import scipy.optimize
import scipy.signal
import scipy.stats
import six
import sklearn.ensemble


__all__ = (
    "SNVCollection",
    "CNACollection",
    "Solver",
)


InferCommand = collections.namedtuple("InferCommand", ["desc", "func"])

_LOGGER = logbook.Logger("LiCl")
_LOG_FORMAT = "[{record.level_name}] {record.channel}: {record.message}"

_DEFAULT_CELLULARITY = 0.8

GENOME = pandas.DataFrame(
    {
        "length": [
            249250621,
            243199373,
            198022430,
            191154276,
            180915260,
            171115067,
            159138663,
            146364022,
            141213431,
            135534747,
            135006516,
            133851895,
            115169878,
            107349540,
            102531392,
            90354753,
            81195210,
            78077248,
            59128983,
            63025520,
            48129895,
            51304566,
            155270560,
            59373566,
            16569,
        ],
    },
    index=pandas.Index(
        [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "X",
            "Y",
            "MT",
        ],
        name="chromosome",
    ),
)


class InferenceError(RuntimeError):
    pass


class SNVCollection(object):
    """A collection of SNV records.

    Information of single nucleotide variations (SNVs) includes the
    positions, the bases, the read counts of the variations.

    Attributes
    ----------
    data : pandas.DataFrame
        The SNV table.
    annotations : pandas.DataFrame
        The annotation table.
    """

    __slots__ = "data", "sample_data", "annotations"
    _ORIG_COLUMNS = ["CHROM", "POS", "REF", "ALT"]
    _BASE_COLUMNS = ["chromosome", "start", "reference", "alternative"]

    @property
    def samples(self):
        """list[str] : All samples in this collection."""
        return self.sample_data.columns.levels[0]

    @property
    def index(self):
        """pandas.Index: Indices of all samples in this collection."""
        return self.data.index

    @property
    def is_female(self):
        """bool : Whether any SNPs available in the Y chromosome."""
        return "Y" not in self.data["chromosome"].values

    def __init__(self, data=None, sample_data=None, annotations=None, *args,
                 **kargs):
        """Create a SNVCollection object.

        Parameters
        ----------
        data : pandas.DataFrame
            The SNV table.
        sample_data : pandas.DataFrame
            The sample table.
        annotations : pandas.DataFrame
            The annotation table.
        """
        if data is None:
            data = pandas.DataFrame(
                columns=pandas.MultiIndex.from_tuples(self._BASE_COLUMNS))
            data["chromosome"] = data["chromosome"].astype(str)
            data["start"] = data["start"].astype("i4")
            data["reference"] = data["reference"].astype(str)
            data["alternative"] = data["alternative"].astype(str)
            data.index.name = "snv"
        if sample_data is None:
            sample_data = pandas.DataFrame(index=data.index)
            sample_data.index.name = "snv"
        if annotations is None:
            annotations = pandas.DataFrame(index=data.index)
            annotations.index.name = "snv"
        self.data = data
        self.sample_data = sample_data
        self.annotations = annotations
        if self.is_female:
            _LOGGER.info("The subject is considered as female.")
        else:
            _LOGGER.info("The subject is considered as male.")

    def __len__(self):
        """Get the number of SNVs."""
        return len(self.data)

    @classmethod
    def from_vcf(cls, vcf):
        """Load SNVs from the DREAM SMC-Het Challenge VCF file.

        Parameters
        ----------
        vcf : file-like object, pathlib.Path or str
            A path or a file handle of the VCF file.

        Returns
        -------
        coll : licl.SNVCollection
            The SNV collection read from the SNV file.
        """
        columns = cls._read_vcf_columns(vcf)
        orig = pandas.read_table(vcf, comment="#", header=None, names=columns,
                                 index_col=False)
        data = cls._read_data(orig)
        sample_data = cls._read_sample_data(orig)
        annotations = pandas.DataFrame({"dbsnp": orig["ID"].values},
                                       index=orig.index)
        return cls(data, sample_data, annotations)

    @classmethod
    def from_truth(cls, truth_vcf, truth_2a=None):
        """Load SNVs from the DREAM SMC-Het Challenge truth file.

        Besides common information present in the VCF file, it also
        loads annotations of whether SNVs are false positive. If given
        Subchallenge 2A truth file, it can also load SNV clustering
        annotations.

        Parameters
        ----------
        truth_vcf : file-like object, pathlib.Path or str
            A path or a file handle of the VCF file.
        truth_2a : file-like object, pathlib.Path or str, default=None
            An optional Subchallenge 2A truth file.

        Returns
        -------
        coll : licl.SNVCollection
            The SNV collection read from the file(s).
        """
        coll = cls.from_vcf(truth_vcf)
        truth_column = 11 if "tumour" in coll.samples else 10
        coll.annotations["truth"] = pandas.read_table(
            truth_vcf, comment="#", header=None, usecols=[truth_column])
        if truth_2a is not None:
            coll.annotations["subclone"] = (
                coll.annotations["truth"].astype("i1"))
            coll.annotations.loc[coll.annotations["truth"], "subclone"] = (
                numpy.loadtxt(truth_2a).astype("i1"))
        return coll

    @classmethod
    def _read_vcf_columns(cls, vcf):
        with open(vcf) as vcf_file:
            for line in vcf_file:
                if line[1] != "#" and line[0] == "#":
                    columns = line[1:].strip().split()
                    return columns
        assert False, "VCF header line not found"

    @classmethod
    def _read_data(cls, orig):
        data = copy.deepcopy(orig.loc[:, cls._ORIG_COLUMNS])
        data.columns = cls._BASE_COLUMNS
        data["chromosome"] = data["chromosome"].astype(str)
        data["start"] = data["start"].astype("i4")
        data["reference"] = data["reference"].astype(str)
        data["alternative"] = data["alternative"].astype(str)
        data.index.name = "snv"
        return data

    @classmethod
    def _read_sample_data(cls, orig):
        sample_data = pandas.DataFrame(index=orig.index)
        for sample_name in orig.columns[9:]:
            sample = orig[sample_name].str.split(":", expand=True)
            if sample_name == "tumor":
                sample_name = "tumour" ## AE spelling hatred
            sample_data[sample_name, "total"] = sample[3].astype("i4")
            sample_data[sample_name, "vaf_3digit"] = sample[4].astype("f4")
            read_counts = sample[1].str.split(",", expand=True).astype("i4")
            sample_data[sample_name, "reference"] = read_counts[0]
            sample_data[sample_name, "alternative"] = read_counts[1]
            sample_data[sample_name, "vaf"] = (
                read_counts[1] / read_counts.sum(axis=1)).astype("f4")
            if sample_name == "tumour":
                sample_data["tumour", "phred"] = sample[2].astype("i1")
        sample_data.columns = pandas.MultiIndex.from_tuples(sample_data.columns)
        return sample_data


class CNACollection(object):
    """A collection of CNA records.

    Information of copy number aberrations (CNAs) includes whether
    variations are 1-state or 2-state in the tumour samples, B-allele
    frequencies, and estimated copy numbers.

    Attributes
    ----------
    data : pandas.DataFrame
        The CNA table
    solutions : pandas.DataFrame
        The CNA solution table
    """

    __slots__ = "data", "solutions"
    _RAW_COLUMNS = slice(0, 4)
    _CNA_COLUMNS = ["chromosome", "start", "end", "baf"]
    _EXTRA_COLUMNS = ["state_count"]
    _RAW_SOLUTION_COLUMNS = [0, 1, 2, 3, 4, 5]
    _SOLUTION_INDICES = ["cna", "solution"]
    _STATE1_COLUMNS = ["major1", "minor1", "frac1"]
    _STATE2_COLUMNS = ["major2", "minor2", "frac2"]
    SOLUTION_COLUMNS = _STATE1_COLUMNS + _STATE2_COLUMNS

    @property
    def trivial_solution(self):
        """pandas.Series : A trivial solution for 2-state CNAs.

        It only pick up the first solution for all CNA record..
        """
        return (self.solutions.reset_index().drop_duplicates("cna")
                .set_index("cna")["solution"])

    def __init__(self, data=None, solutions=None, *args, **kargs):
        """Create a CNACollection object.

        Parameters
        ----------
        data : pandas.DataFrame
            The CNA table.
        solutions : pandas.DataFrame
            The solution table.
        """
        if data is None:
            data = pandas.DataFrame(
                columns=(self._CNA_COLUMNS + self._EXTRA_COLUMNS))
            data["chromosome"] = data["chromosome"].astype(str)
            data["start"] = data["start"].astype("i4")
            data["end"] = data["end"].astype("i4")
            data["baf"] = data["baf"].astype("f4")
            data["state_count"] = data["state_count"].astype("i1")
            data.index.name = "cna"
        if solutions is None:
            solutions = pandas.DataFrame(
                index=data.index, columns=self.SOLUTION_COLUMNS, dtype="f4")
        self.data = data
        self.solutions = solutions

    def enforce_isa(self, inplace=False):
        """Enforce Infinite Site Assumptions (ISA).

        Parameters
        ----------
        inplace : bool, default=False
            Whether to apply operations inplace.

        Returns
        -------
        coll : licl.CNACollections
            Cleaned CNA records.
        """
        if not inplace:
            coll = copy.copy(self)
            return coll.enforce_isa(inplace=True)
        else:
            coll = self
        valid_solutions = (
            ((coll.solutions["major1"] == 1.0) &
             (coll.solutions["minor1"] == 1.0)) |
            ((coll.solutions["major2"] == 1.0) &
             (coll.solutions["minor2"] == 1.0))
        )
        coll.solutions = copy.deepcopy(coll.solutions[valid_solutions])
        valid_cna_idx = coll.solutions.index.get_level_values("cna").unique()
        coll.data = copy.deepcopy(coll.data.loc[valid_cna_idx, :])
        coll._sort_solutions()
        return coll

    def extend(self, inplace=False):
        """Extend all CNAs to cover their whole chromosomes.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to apply operations inplace.

        Returns
        -------
        coll : licl.CNACollections
            Extended CNA records.
        """
        coll = self if inplace else copy.deepcopy(self)
        last_idx = None
        for idx, cna in coll.data.iterrows():
            if last_idx is None:
                coll.data.at[idx, "start"] = 1
                last_idx = idx
                continue
            if cna["chromosome"] != coll.data.at[last_idx, "chromosome"]:
                coll.data.at[idx, "start"] = 1
                last_chrom = coll.data.at[last_idx, "chromosome"]
                chrom_len = GENOME.at[last_chrom, "length"]
                coll.data.at[last_idx, "end"] = chrom_len + 1
                last_idx = idx
                continue
            breakpoint = (cna["start"] + coll.data.at[last_idx, "end"]) // 2
            coll.data.at[last_idx, "end"] = breakpoint
            coll.data.at[idx, "start"] = breakpoint
            last_idx = idx
        if last_idx is not None:
            last_chrom = coll.data.at[last_idx, "chromosome"]
            chrom_len = GENOME.at[last_chrom, "length"]
            coll.data.at[last_idx, "end"] = chrom_len + 1
        _LOGGER.debug("Extended CNA:\n{}", coll.data)
        return coll

    def match(self, snvs, annotate_snv=False):
        """Match the SNVs to the CNA records.

        Assign SNVs to their corresponding CNA regions.

        Parameters
        ----------
        snvs : licl.SNVCollection
            SNV records to match.

        Returns
        -------
        cnas : pandas.Series
            A list of matched CNA indices.
        """
        cnas = pandas.Series(-1, name="cna", index=snvs.index, dtype="i4")
        for chrom in snvs.data["chromosome"].unique():
            chrom_snvs_check = snvs.data["chromosome"] == chrom
            chrom_snvs = snvs.data[chrom_snvs_check].reset_index()
            chrom_cnas_check = self.data["chromosome"] == chrom
            chrom_cnas = self.data[chrom_cnas_check].reset_index()
            merged = chrom_snvs.merge(chrom_cnas, how="inner", on="chromosome")
            merged = merged[(merged["start_x"] >= merged["start_y"]) &
                            (merged["start_x"] < merged["end"])]
            merged.drop_duplicates("snv", inplace=True)
            merged.set_index("snv", inplace=True)
            cnas.loc[merged.index] = merged["cna"]
        if annotate_snv:
            snvs.annotations["cna"] = cnas
        return cnas

    def solve(self, solution=None):
        """Solve all CNAs using the given solution.

        Parameters
        ----------
        solution : array-like, default=None
            The solution solution of all CNAs. Its length should be the
            same as self.data. If it is None, self.trivial_solution is
            applied instead.

        Returns
        -------
        merged : pandas.DataFrame
            Solved CNA records.
        """
        if solution is None:
            solution = self.trivial_solution
        data = self.data.reset_index()
        data["solution"] = solution
        merged = data.merge(self.solutions, how="left",
                            left_on=["cna", "solution"], right_index=True)
        return merged.drop("solution", axis=1).set_index("cna")

    @classmethod
    def from_battenberg(cls, bat):
        """Load CNAs from the Battenberg file.

        Parameters
        ----------
        bat : file-like object, pathlib.Path or str
            A path or a file handle of the Battenberg file.

        Returns
        -------
        coll : licl.SNVCollection
            The SNV collection read from the file(s).
        """
        coll = cls()
        orig = pandas.read_table(bat)
        cls._read_cnas(coll, orig)
        cls._read_solutions(coll, orig)
        cls._update_state_counts(coll)
        return coll

    def _sort_solutions(self):
        check = ((self.solutions["major2"] != 1.0) |
                 (self.solutions["minor2"] != 1.0))
        old_state1 = copy.deepcopy(
            self.solutions.loc[check, self._STATE1_COLUMNS])
        old_state2 = copy.deepcopy(
            self.solutions.loc[check, self._STATE2_COLUMNS])
        self.solutions.loc[check, self._STATE1_COLUMNS] = old_state2.values
        self.solutions.loc[check, self._STATE2_COLUMNS] = old_state1.values

    @classmethod
    def _read_cnas(cls, coll, orig):
        coll.data = copy.deepcopy(orig.iloc[:, cls._RAW_COLUMNS])
        coll.data.columns = cls._CNA_COLUMNS
        coll.data["chromosome"] = coll.data["chromosome"].astype(str)
        coll.data["start"] = coll.data["start"].astype("i4")
        coll.data["end"] = coll.data["end"].astype("i4")
        coll.data["baf"] = coll.data["baf"].astype("f4")
        coll.data["start"] += 1
        coll.data["end"] += 1
        coll.data.index.name = "cna"

    @classmethod
    def _read_solutions(cls, coll, orig):
        sols = list()
        for j in range(6):
            cols = [j * 10 + 7 + k for k in cls._RAW_SOLUTION_COLUMNS]
            sol = orig.iloc[:, cols]
            sol.columns = cls.SOLUTION_COLUMNS
            sol.index.name = "cna"
            sol = sol[~sol["major1"].isnull()]
            sol.insert(0, "solution", j)
            sols.append(sol.reset_index())
        coll.solutions = pandas.concat(sols)
        coll.solutions.reset_index(drop=True, inplace=True)
        coll.solutions.set_index(["cna", "solution"], inplace=True)
        coll.solutions.sort_index(inplace=True)
        coll.solutions.fillna({"major2": 1.0, "minor2": 1.0, "frac2": 0.0},
                              inplace=True)

    @classmethod
    def _update_state_counts(cls, coll):
        coll.data["state_count"] = 2
        state1 = coll.solutions[coll.solutions["frac1"] == 1.0]
        index_state1 = state1.index.get_level_values("cna")
        coll.data.loc[index_state1, "state_count"] = 1
        state0 = coll.solutions[(coll.solutions["frac1"] == 1.0) &
                                (coll.solutions["major1"] == 1.0) &
                                (coll.solutions["minor1"] == 1.0)]
        index_state0 = state0.index.get_level_values("cna")
        coll.data.loc[index_state0, "state_count"] = 0


class BaseSolverComponent(object):
    """The abstract solver component."""

    def __init__(self, snvs, cnas, precalc=None, exclude=None,
                 *args, **kargs):
        """Create a BaseSolver object.

        Parameters
        ----------
        snvs : licl.SNVCollection
            All SNV records for inference.
        cnas : licl.CNACollection
            All CNA records for inference.
        precalc: pathlib.Path or str-like, default=None
            A pre-calculated file that summarises average ploidy and
            tumour purity.
        exclude : list[str], default=[]
            Data that should be excluded during training.
        """
        self._snvs = snvs
        self._cnas = cnas
        self._precalc = precalc
        if exclude is None:
            exclude = []
        self._exclude = exclude

    @property
    def snvs(self):
        """licl.SNVCollection : All SNV records for inference."""
        return self._snvs

    @property
    def cnas(self):
        """licl.CNACollection: All CNA records for inference."""
        return self._cnas

    @property
    def precalc(self):
        """pathlib.Path or str-like, default=None : A cellularity file
        that summarises average ploidy and tumour purity.
        """
        return self._precalc

    @property
    def exclude(self):
        """list[str] : Data that should be excluded during training."""
        return self._exclude


@six.add_metaclass(abc.ABCMeta)
class BaseFalsePositiveSNVPredictor(BaseSolverComponent):
    """The false positive SNV predictor interface."""

    def __init__(self, *args, **kargs):
        """Create a BaseFalsePositiveSNVPredictor object."""
        super(BaseFalsePositiveSNVPredictor, self).__init__(*args, **kargs)
        self._predictions = None

    @property
    def predictions(self):
        """pandas.Series[bool] : False positive SNV predictions."""
        if self._predictions is None:
            self._predict()
        return self._predictions

    @abc.abstractmethod
    def _predict(self):
        pass


class DummySNVPredictor(BaseFalsePositiveSNVPredictor):
    """A filter that accepts all SNVs."""

    def _predict(self):
        self._predictions = pandas.Series(True, index=self.snvs.index,
                                          name="predictions")


class KnownSNVFilter(BaseFalsePositiveSNVPredictor):
    """A filter that identifies known false positive SNVs."""

    def __init__(self, *args, **kargs):
        """Create a KnownSNVFilter object.

        Parameters
        ----------
        snvs : licl.SNVCollection
            All SNV records for inference.
        cnas : licl.CNACollection
            All CNA records for inference.
        precalc: pathlib.Path or str-like, default=None
            A pre-calculated file that summarises average ploidy and
            tumour purity.
        exclude : list[str], default=[]
            Data that should be excluded during training.
        """
        super(KnownSNVFilter, self).__init__(*args, **kargs)
        known_false_snvs = list()
        path = pkg_resources.resource_filename("licl", "data")
        _LOGGER.debug("Loading known annotations from {}", path)
        for sample in os.listdir(path):
            sample_dir = os.path.join(path, sample)
            for sample_file in os.listdir(sample_dir):
                if not sample_file.endswith(".truth.scoring_vcf.vcf"):
                    continue
                vcf_path = os.path.join(sample_dir, sample_file)
                _LOGGER.debug("Loading {} ...", vcf_path)
                vcf = SNVCollection.from_truth(vcf_path)
                false_snvs = vcf.data[~vcf.annotations["truth"]]
                _LOGGER.debug("Loaded {} false SNV(s).", len(false_snvs))
                known_false_snvs.append(false_snvs)
        known_false_snvs = pandas.concat(known_false_snvs)
        known_false_snvs.drop_duplicates(["chromosome", "start"], inplace=True)
        _LOGGER.info("Total known false SNVs: {}", len(known_false_snvs))
        known_false_snvs.set_index(["chromosome", "start"], inplace=True)
        self._known_false_snvs = self._exclude_data_point(known_false_snvs)
        _LOGGER.info("After exclusion, {} known false SNV(s) remained.",
                     len(self._known_false_snvs))

    def _exclude_data_point(self, data):
        path = pkg_resources.resource_filename("licl", "data")
        for sample in self.exclude:
            if sample not in os.listdir(path):
                continue
            sample_dir = os.path.join(path, sample)
            for sample_file in os.listdir(sample_dir):
                if not sample_file.endswith(".truth.scoring_vcf.vcf"):
                    continue
                vcf_path = os.path.join(sample_dir, sample_file)
                _LOGGER.debug("Excluding {} ...", vcf_path)
                vcf = SNVCollection.from_truth(vcf_path)
                false_snvs = vcf.data[~vcf.annotations["truth"]]
                false_snvs.set_index(["chromosome", "start"], inplace=True)
                orig_len = len(data)
                data.drop(false_snvs.index, inplace=True, errors="ignore")
                _LOGGER.debug("Excluded {} known false SNV(s).",
                              orig_len - len(data))
        return data

    def _predict(self):
        merged = self.snvs.data.reset_index().merge(
            self._known_false_snvs, how="inner",
            left_on=["chromosome", "start"], right_index=True)
        _LOGGER.info("Identified {} known false SNV(s).", len(merged))
        self._predictions = pandas.Series(True, index=self.snvs.index,
                                          name="predictions")
        self._predictions[merged["snv"].values] = False


class RandomForestSNVPredictor(BaseFalsePositiveSNVPredictor):
    """A false positive SNV predictor using random forest classifier."""

    def __init__(self, *args, **kargs):
        """Create a RandomForestSNVPredictor object.

        Parameters
        ----------
        snvs : licl.SNVCollection
            All SNV records for inference.
        cnas : licl.CNACollection
            All CNA records for inference.
        precalc: pathlib.Path or str-like, default=None
            A pre-calculated file that summarises average ploidy and
            tumour purity.
        exclude : list[str], default=[]
            Data that should be excluded during training.
        """
        super(RandomForestSNVPredictor, self).__init__(*args, **kargs)
        train = list()
        path = pkg_resources.resource_filename("licl", "data")
        _LOGGER.debug("Loading training data from: {}", path)
        for sample in os.listdir(path):
            sample_dir = os.path.join(path, sample)
            for sample_file in os.listdir(sample_dir):
                if not sample_file.endswith(".truth.scoring_vcf.vcf"):
                    continue
                vcf_path = os.path.join(sample_dir, sample_file)
                _LOGGER.debug("Loading {} ...", vcf_path)
                vcf = SNVCollection.from_truth(vcf_path)
                if "tumour" not in vcf.samples:
                    _LOGGER.debug("Bad VCF truth file {}, skipping...",
                                  vcf_path)
                    continue
                features = pandas.concat(
                    [
                        vcf.annotations["truth"],
                        vcf.data["chromosome"],
                        vcf.data["start"],
                        vcf.annotations["dbsnp"] == ".",
                        vcf.sample_data["tumour", "vaf"],
                        vcf.sample_data["normal", "vaf"],
                        vcf.sample_data["normal", "alternative"],
                        ],
                    axis=1,
                )
                features.set_index(["chromosome", "start"])
                train.append(features)
        train = pandas.concat(train)
        _LOGGER.info("Loaded {} data point(s).", len(train))
        train.set_index(["chromosome", "start"], inplace=True)
        train = self._exclude_data_points(train)
        _LOGGER.info("Building a random forest model...")
        self._model = sklearn.ensemble.RandomForestClassifier(300)
        self._model.fit(train.iloc[:, 1:].values, train.iloc[:, 0].values)
        _LOGGER.info("Prediction model is built.")

    def _exclude_data_points(self, data):
        path = pkg_resources.resource_filename("licl", "data")
        for sample in self.exclude:
            if sample not in os.listdir(path):
                continue
            sample_dir = os.path.join(path, sample)
            for sample_file in os.listdir(sample_dir):
                if not sample_file.endswith(".truth.scoring_vcf.vcf"):
                    continue
                vcf_path = os.path.join(sample_dir, sample_file)
                _LOGGER.debug("Excluding {} ...", vcf_path)
                vcf = SNVCollection.from_truth(vcf_path)
                if "tumour" not in vcf.samples:
                    continue
                orig_len = len(data)
                data.drop(vcf.data[["chromosome", "start"]].values,
                          inplace=True)
                _LOGGER.debug("Excluded {} data point(s).",
                              orig_len - len(data))
        _LOGGER.info("After exclusion, {} data point(s) remained.", len(data))
        data.reset_index(drop=True, inplace=True)
        return data

    def _predict(self):
        target = pandas.concat(
            [
                self.snvs.annotations["dbsnp"] == ".",
                self.snvs.sample_data["tumour", "vaf"],
                self.snvs.sample_data["normal", "vaf"],
                self.snvs.sample_data["normal", "alternative"],
                ],
            axis=1,
        )
        self._predictions = pandas.Series(self._model.predict(target.values),
                                          index=self.snvs.index,
                                          name="predictions")
        _LOGGER.info("Predicted {} false SNV(s).", (~self._predictions).sum())


class CombinedSNVPredictor(BaseFalsePositiveSNVPredictor):
    """A combined SNV predictor."""

    def __init__(self, *args, **kargs):
        """Create a CombinedSNVPredictor object.

        Parameters
        ----------
        snvs : licl.SNVCollection
            All SNV records for inference.
        cnas : licl.CNACollection
            All CNA records for inference.
        precalc: pathlib.Path or str-like, default=None
            A pre-calculated file that summarises average ploidy and
            tumour purity.
        exclude : list[str], default=[]
            Data that should be excluded during training.
        """
        super(CombinedSNVPredictor, self).__init__(*args, **kargs)
        self._known_filter = KnownSNVFilter(*args, **kargs)
        self._predictor = RandomForestSNVPredictor(*args, **kargs)

    def _predict(self):
        filtered = self._known_filter.predictions
        predicted = self._predictor.predictions
        table = numpy.asarray(
            [
                [(filtered & predicted).sum(),
                 (filtered & ~predicted).sum(),],
                [(~filtered & predicted).sum(),
                 (~filtered & ~predicted).sum(),],
            ],
            dtype="i8",
        )
        p_value = scipy.stats.fisher_exact(table)[1]
        if p_value < 0.05:
            _LOGGER.info("Combined known SNV list and prediction.")
            pred = filtered & predicted
        else:
            _LOGGER.warning("Known SNVs seem not helpful in this dataset.")
            pred = predicted
        _LOGGER.notice("Annotated {} false positive SNV(s).", (~pred).sum())
        self._predictions = pred


@six.add_metaclass(abc.ABCMeta)
class BaseCellularityEstimator(BaseSolverComponent):
    """The cellularity estimating interface."""

    @abc.abstractproperty
    def reliable(self):
        """bool : Whether the result is reliable or not."""
        pass

    @abc.abstractproperty
    def cellularity(self):
        """float : The estimated cellularity."""
        pass


class RandomGuessCellularityEstimator(BaseCellularityEstimator):
    """An estimator that always return a default cellularity."""

    @property
    def reliable(self):
        """bool(False) : The estimation is not reliable."""
        return False

    @property
    def cellularity(self):
        """float : The default cellularity."""
        return _DEFAULT_CELLULARITY


class CellularityFileParser(BaseCellularityEstimator):
    """The cellularity file parser."""

    @property
    def reliable(self):
        """bool(True) : The estimation is reliable."""
        return True

    @property
    def cellularity(self):
        """float : The estimated cellularity."""
        if self.precalc is None:
            raise ValueError("no precalculated file")
        _LOGGER.info("Read pre-calculated cellularity.")
        with open(self.precalc, "rt") as source:
            title = source.readline().strip().split()
            try:
                position = title.index("cellularity")
            except ValueError:
                position = 0
            values = source.readline().strip().split()
            result = float(values[position])
        _LOGGER.notice("Cellularity estimation method: File.")
        _LOGGER.notice("Estimated cellularity: {:f}", result)
        return result


class BattenbergCellularityEstimator(BaseCellularityEstimator):
    """A battenberg cellularity estimator."""

    @property
    def reliable(self):
        """bool(True) : The estimation is reliable."""
        return True

    def __init__(self, *args, **kargs):
        """Create a BattenbergCellularityEstimator object.

        Parameters
        ----------
        snvs : licl.SNVCollection
            All SNV records for inference.
        cnas : licl.CNACollection
            All CNA records for inference.
        precalc: pathlib.Path or str-like, default=None
            A pre-calculated file that summarises average ploidy and
            tumour purity.
        exclude : list[str], default=[]
            Data that should be excluded during training.
        only_2state : bool
            Whether to calculate cellularity based on only 2-state CNA
            records.
        """
        super(BattenbergCellularityEstimator, self).__init__(*args, **kargs)
        self._only_2state = True

    @property
    def cellularity(self):
        """float : The estimated cellularity."""
        _LOGGER.info("Use CNA records to estimate cellularity.")
        cnas = self.cnas.enforce_isa().solve()
        cnas = cnas[(cnas["major1"] != cnas["minor1"]) |
                    (cnas["major2"] != cnas["minor2"])]
        if self._only_2state:
            _LOGGER.info("Filtering 1-state CNA records...")
            check = cnas["state_count"] == 2
            if check.any():
                cnas = cnas[check]
            else:
                _LOGGER.warning("No CNA records left after filtering. Falling "
                                "back to use all records.")
        _LOGGER.debug("CNA records:\n{}\n", cnas)
        results = (
            (1 - 2 * cnas["baf"]) /
            (cnas["baf"] * cnas["frac1"] * (cnas["major1"] + cnas["minor1"]) +
             cnas["baf"] * cnas["frac2"] * (cnas["major2"] + cnas["minor2"]) +
             1 - 2 * cnas["baf"] - cnas["major1"] * cnas["frac1"] -
             cnas["major2"] * cnas["frac2"])
        )
        results = results[~results.isnull()]
        _LOGGER.debug("Cellularities:\n{}\n", results)
        if len(results) == 0:
            _LOGGER.warning("Unable to infer cellularity from Battenberg: "
                            "major and minor allele copy numbers are same.")
            raise ValueError("major and minor allele copy numbers are same")
        median = results.median()
        _LOGGER.info("Median cellularity: {:f}", median)
        regular_values = results[(results - median).abs() < 0.1]
        if len(regular_values) == 0:
            result = results.mean()
            _LOGGER.info("Raw mean cellularity: {:f}", result)
        else:
            result = regular_values.mean()
            _LOGGER.info("Filtered mean cellularity: {:f}", result)
        if result > 1.0:
            _LOGGER.info("Raw mean cellularity is greater than 1.0")
            _LOGGER.info("Reduce it to 1.0")
            result = 1.0
        elif result < 0.0:
            _LOGGER.info("Raw mean cellularity is smaller than 0.0")
            _LOGGER.info("Increase it to 0.0")
            result = 0.0
        _LOGGER.notice("Cellularity estimation method: Battenberg.")
        _LOGGER.notice("Estimated cellularity: {:f}", result)
        return result


class AllosomeCellularityEstimator(BaseCellularityEstimator):
    """The allosome cellularity estimator."""

    _GRID_COUNT = 100

    @property
    def reliable(self):
        """bool(False) : The estimation is not reliable."""
        return False

    @property
    def cellularity(self):
        """float : The estimated cellularity"""
        vaf = self.snvs.sample_data.loc[
            ((self.snvs.data["chromosome"] == "X") |
             (self.snvs.data["chromosome"] == "Y")),
            ("tumour", "vaf")
        ]
        kde = scipy.stats.gaussian_kde(vaf.values)
        delta = 0.5 / self._GRID_COUNT
        x = numpy.linspace(0.0 + delta, 1.0 - delta, self._GRID_COUNT)
        log_dens = kde.logpdf(x)
        if log_dens[-1] > log_dens[-2]:
            _LOGGER.notice("Cellularity estimation method: Allosome SNVs.")
            _LOGGER.notice("Estimated cellularity: {:f}", self._cellularity)
            return 1.0
        max_peak = x[scipy.signal.argrelmax(log_dens)[0][-1]]
        _LOGGER.debug("Maximal peak found at {:f}.", max_peak)
        result = scipy.optimize.fminbound(
            lambda x: -kde.logpdf(x), max_peak - delta, max_peak + delta,
            xtol=1e-6, disp=0)[0]
        _LOGGER.notice("Cellularity estimation method: Allosome SNVs.")
        _LOGGER.notice("Estimated cellularity: {:f}", result)
        return result


class CombinedCellularityEstimator(BaseCellularityEstimator):
    """The combined cellularity estimator."""

    def __init__(self, *args, **kargs):
        """Create a CombinedCellularityEstimator object.

        Parameters
        ----------
        snvs : licl.SNVCollection
            All SNV records for inference.
        cnas : licl.CNACollection
            All CNA records for inference.
        precalc: pathlib.Path or str-like, default=None
            A pre-calculated file that summarises average ploidy and
            tumour purity.
        exclude : list[str], default=[]
            Data that should be excluded during training.
        """
        super(CombinedCellularityEstimator, self).__init__(*args, **kargs)
        self._estimator = None
        self._result = None

    @property
    def reliable(self):
        """bool : Whether the result is reliable or not."""
        if self._estimator is None:
            self._estimate()
        return self._estimator.reliable

    @property
    def cellularity(self):
        """float : The estimated cellularity"""
        if self._result is None:
            self._estimate()
        return self._result

    def _estimate(self):
        if self.precalc is not None:
            self._estimator = CellularityFileParser(
                self.snvs, self.cnas, self.precalc, self.exclude)
        else:
            cnas = self.cnas.enforce_isa().solve()
            cnas = cnas[(cnas["major1"] != cnas["minor1"]) |
                        (cnas["major2"] != cnas["minor2"])]
            if len(cnas) != 0:
                self._estimator = BattenbergCellularityEstimator(
                    self.snvs, self.cnas, self.precalc, self.exclude)
            else:
                self._estimator = AllosomeCellularityEstimator(
                    self.snvs, self.cnas, self.precalc, self.exclude)
        self._result = self._estimator.cellularity


@six.add_metaclass(abc.ABCMeta)
class BaseSubcloneEstimator(BaseSolverComponent):
    """A subclone estimator interface."""

    def __init__(self, snv_predictor, cellularity_estimator, *args, **kargs):
        """Create a BaseSubcloneEstimator object.

        Parameters
        ----------
        snv_predictor : licl.BaseFalsePositiveSNVPredictor
            A false positive SNV predictor.
        cellularity_estimator : licl.BaseCellularityEstimator
            A cellularity estimator.
        snvs : licl.SNVCollection
            All SNV records for inference.
        cnas : licl.CNACollection
            All CNA records for inference.
        precalc: pathlib.Path or str-like, default=None
            A pre-calculated file that summarises average ploidy and
            tumour purity.
        exclude : list[str], default=[]
            Data that should be excluded during training.
        """
        super(BaseSubcloneEstimator, self).__init__(*args, **kargs)
        self._snv_predictor = snv_predictor
        self._cellularity_estimator = cellularity_estimator
        self._subclones = None

    @property
    def subclones(self):
        """pandas.DataFrame: A table of probabilities of individual
        SNVs being assigned to each subclone. The column titles are
        the subclonal prevalences."""
        if self._subclones is None:
            self._infer()
            self._subclones.sort_index(axis=1, ascending=False)
        return self._subclones

    @abc.abstractmethod
    def _infer(self):
        pass


class SingleSubcloneEstimator(BaseSubcloneEstimator):
    """A subclone estimator that always report one subclone."""

    def _infer(self):
        self._subclones = pandas.DataFrame(
            {self._cellularity_estimator.cellularity: 1.0},
            index=self._snvs.index,
        )


@six.add_metaclass(abc.ABCMeta)
class BaseDensityBasedSubcloneEstimator(BaseSubcloneEstimator):
    """A density-based subclone estimator class with a helper function."""

    _GRID_COUNT = 100

    def __init__(self, *args, **kargs):
        """Create a BaseDensityBasedSubcloneEstimator object."""
        super(BaseDensityBasedSubcloneEstimator, self).__init__(*args, **kargs)
        self._matched_snvs = None

    def _match_snv_to_isa_cna(self):
        if self._matched_snvs is None:
            extended_isa_cna = self.cnas.enforce_isa().extend()
            cna_assign = extended_isa_cna.match(self.snvs)
            self._matched_snvs = self.snvs.data.join(cna_assign).merge(
                extended_isa_cna.solve(), how="left",
                left_on="cna", right_index=True)
        return self._matched_snvs

    def _get_density_peaks(self, samples):
        kde = scipy.stats.gaussian_kde(samples.flatten())
        delta = 0.5 / self._GRID_COUNT
        x = numpy.linspace(0.0 + delta, 1.0 - delta, self._GRID_COUNT)
        log_dens = kde.logpdf(x)
        maxidx = numpy.unique(numpy.hstack((
            [0] if log_dens[0] > log_dens[1] else [],
            scipy.signal.argrelmax(log_dens)[0],
            [len(log_dens) - 1] if log_dens[-1] > log_dens[-2] else [],
        )).astype(int))
        peaks = x[maxidx]
        for j in range(peaks.size):
            peaks[j] = scipy.optimize.fminbound(
                lambda x: -kde.logpdf(x), peaks[j] - delta, peaks[j] + delta,
                xtol=1e-6, full_output=False, disp=False)
        peaks.sort()
        peaks = peaks[::-1]
        _LOGGER.debug("Found raw peaks: {}", peaks)
        return peaks

    def _adjust_peaks_using_cellularity(self, peaks):
        if self._cellularity_estimator.reliable:
            max_frac = self._cellularity_estimator.cellularity
            diff = numpy.abs(peaks - max_frac)
            max_peak_idx = diff.argmin()
            if diff[max_peak_idx] > 0.1:
                peaks = numpy.hstack((peaks[peaks < max_frac], max_frac))
            else:
                peaks = peaks[peaks <= peaks[max_peak_idx]]
        max_peak = peaks.max()
        if max_peak >= 1.1:
            peaks = peaks[peaks < 1.1]
            max_peak = peaks.max()
        if max_peak >= 1:
            peaks = numpy.hstack((peaks[peaks < 1.0], 1.0))
        peaks.sort()
        peaks = peaks[::-1]
        _LOGGER.info("Found peaks: {}", peaks)
        return peaks

    def _update_subclones(self, samples, peaks):
        _LOGGER.debug("Found raw subclone prevalences: {}", peaks)
        weights = self._estimate_weights(samples, peaks)
        while True:
            scores = self._assign_subclones(peaks, weights)
            true_scores = scores.loc[self._snv_predictor.predictions, :]
            assignment = true_scores.values.argmax(axis=1)
            valid_idx, size = numpy.unique(assignment, return_counts=True)
            valid_idx = valid_idx[size >= 25]
            if valid_idx.size == peaks.size or size.sum() < 100:
                self._subclones = scores
                break
            peaks = peaks[valid_idx]
            weights = self._estimate_weights(samples, peaks)

    @abc.abstractmethod
    def _estimate_weights(self, samples, peaks):
        pass

    def _assign_subclones(self, peaks, weights):
        peaks = peaks[None, :, None]
        merged = self._match_snv_to_isa_cna()
        extended_cna = self.cnas.extend()
        all_matched_snvs = extended_cna.match(self.snvs)
        all_merged = self.snvs.data.join(all_matched_snvs).merge(
            extended_cna.solve(), how="left", left_on="cna", right_index=True)
        if not self.snvs.is_female:
            allosome = ((merged["chromosome_x"] == "X") |
                        (merged["chromosome_x"] == "Y"))
            merged.loc[allosome, ["major1", "frac1", "major2", "minor2"]] = 1.0
            merged.loc[allosome, ["minor1", "frac2"]] = 0.0
        to_fill = merged["frac1"].isnull()
        merged.fillna(all_merged.loc[to_fill, CNACollection.SOLUTION_COLUMNS],
                      axis=1, inplace=True)
        missed = merged[CNACollection.SOLUTION_COLUMNS].isnull().any(axis=1)
        cellularity = peaks.max()
        major = merged["major1"].values[:, None, None]
        minor = merged["minor1"].values[:, None, None]
        frac = merged["frac1"].values[:, None, None]
        denom = (frac * cellularity * (major + minor) +
                 (1 - frac * cellularity) * 2)
        numer = numpy.dstack((
            numpy.where(peaks < frac, 0, (major * frac * cellularity +
                                          peaks - frac * cellularity)),
            numpy.where(peaks < frac, 0, (minor * frac * cellularity +
                                          peaks - frac * cellularity)),
            numpy.repeat(peaks, len(merged), axis=0),
        ))
        p = numer / denom
        tumour_data = self.snvs.sample_data.loc[merged.index, "tumour"]
        x = tumour_data["alternative"].values[:, None, None]
        k = x + tumour_data["reference"].values[:, None, None]
        llv_sample_subclone = (scipy.stats.binom.logpmf(x, k, p).max(axis=2) +
                               numpy.log(weights))
        llv_sample = scipy.misc.logsumexp(llv_sample_subclone, axis=1)
        return pandas.DataFrame(
            numpy.exp(llv_sample_subclone - llv_sample[:, None]),
            index=merged.index,
            columns=peaks.flatten(),
            dtype="f4",
        )


class State0SubcloneEstimator(BaseDensityBasedSubcloneEstimator):
    """A density-based subclone estimator using non-CNA SNVs."""

    def _infer(self):
        _LOGGER.info("Subclone estimated using SNVs without CNA.")
        samples = self._match_snv_to_isa_cna()
        if self.snvs.is_female:
            check = (self._snv_predictor.predictions &
                     (samples["state_count"] == 0) & ~samples["frac1"].isnull())
        else:
            check = (self._snv_predictor.predictions &
                     (samples["chromosome_x"] != "X") &
                     (samples["chromosome_x"] != "Y") &
                     (samples["state_count"] == 0) & ~samples["frac1"].isnull())
        if check.sum() < 50:
            raise InferenceError(
                "Not enough SNVs in non-CNA regions for subclone inference.")
        samples = samples[check]
        ratios = self.snvs.sample_data.loc[check, ("tumour", "vaf")] * 2
        peaks = self._get_density_peaks(ratios.values)
        peaks = self._adjust_peaks_using_cellularity(peaks)
        self._update_subclones(samples, peaks)

    def _estimate_weights(self, samples, peaks):
        _LOGGER.debug("Found subclone prevalences: {}", peaks)
        peaks = peaks[None, :, None]
        weights = numpy.repeat(1 / peaks.size,peaks.size).astype("f4")
        p = peaks / 2
        tumour_data = self.snvs.sample_data.loc[samples.index, "tumour"]
        x = tumour_data["alternative"].values[:, None, None]
        k = x + tumour_data["reference"].values[:, None, None]
        llv_sample_subclone = ((scipy.stats.binom.logpmf(x, k, p)
                                .max(axis=2)) + numpy.log(weights))
        llv_sample = scipy.misc.logsumexp(llv_sample_subclone, axis=1)
        llv = llv_sample.sum()
        for t in range(1000):
            old_llv = llv
            p_member = numpy.exp(llv_sample_subclone - llv_sample[:, None])
            weights = p_member.mean(axis=0)
            weights /= weights.sum()
            llv_sample_subclone = ((scipy.stats.binom.logpmf(x, k, p)
                                    .max(axis=2)) + numpy.log(weights))
            llv_sample = scipy.misc.logsumexp(llv_sample_subclone, axis=1)
            llv = llv_sample.sum()
            if llv > old_llv and llv < old_llv + 0.001:
                _LOGGER.info("EM is fully converged.")
                break
        else:
            _LOGGER.warning("Maximum iteration reached but the EM is not "
                            "fully converged.")
        return weights


class State1SubcloneEstimator(BaseDensityBasedSubcloneEstimator):
    """A density-based subclone estimator using 1-state-CNA SNVs."""

    def _infer(self):
        _LOGGER.info("Subclone estimated using SNVs with 1-state CNA.")
        samples = self._match_snv_to_isa_cna()
        if self.snvs.is_female:
            check = (self._snv_predictor.predictions &
                     (samples["state_count"] < 2))
        else:
            check = (self._snv_predictor.predictions &
                     (samples["state_count"] < 2) &
                     (samples["chromosome_x"] != "X") &
                     (samples["chromosome_x"] != "Y"))
            allosome_cna = self.cnas.data[self.cnas.data["chromosome"] == "X"]
            if (allosome_cna["state_count"] == 0).all():
                check |= (self._snv_predictor.predictions &
                        ((samples["chromosome_x"] == "X") |
                        (samples["chromosome_x"] == "Y")))
        if check.sum() < 50:
            raise InferenceError("Not enough SNVs in 1-state CNA regions for "
                                 "subclone inference.")
        samples = copy.deepcopy(samples[check])
        if not self.snvs.is_female:
            allosome = ((samples["chromosome_x"] == "X") |
                        (samples["chromosome_x"] == "Y"))
            samples.loc[allosome, ["major1", "frac1", "major2", "minor2"]] = 1
            samples.loc[allosome, ["minor1", "frac2"]] = 0
        vaf = self.snvs.sample_data.loc[samples.index, ("tumour", "vaf")]
        #For samples with CNAs, there will always be a reliable cellularity.
        cellularity = self._cellularity_estimator.cellularity
        ratios = vaf * ((samples["major1"] + samples["minor1"]) * cellularity +
                        2 * (1 - cellularity))
        samples = samples[~ratios.isnull()]
        ratios = ratios[~ratios.isnull()]
        peaks = self._get_density_peaks(ratios.values)
        peaks = self._adjust_peaks_using_cellularity(peaks)
        self._update_subclones(samples, peaks)

    def _estimate_weights(self, samples, peaks):
        _LOGGER.debug("Found subclone prevalences: {}", peaks)
        peaks = peaks[None, :, None]
        cellularity = peaks.max()
        weights = numpy.repeat(1 / peaks.size,peaks.size).astype("f4")
        major = samples["major1"].values[:, None, None]
        minor = samples["minor1"].values[:, None, None]
        denom = (cellularity * (major + minor) + (1 - cellularity) * 2)
        numer = numpy.dstack((
            numpy.where(peaks != peaks.max(), peaks, major * peaks),
            numpy.where(peaks != peaks.max(), peaks, minor * peaks),
        ))
        p = numer / denom
        tumour_data = self.snvs.sample_data.loc[samples.index, "tumour"]
        x = tumour_data["alternative"].values[:, None, None]
        k = x + tumour_data["reference"].values[:, None, None]
        llv_sample_subclone = ((scipy.stats.binom.logpmf(x, k, p)
                                .max(axis=2)) + numpy.log(weights))
        llv_sample = scipy.misc.logsumexp(llv_sample_subclone, axis=1)
        llv = llv_sample.sum()
        for t in range(1000):
            old_llv = llv
            p_member = numpy.exp(llv_sample_subclone - llv_sample[:, None])
            weights = p_member.mean(axis=0)
            weights /= weights.sum()
            llv_sample_subclone = ((scipy.stats.binom.logpmf(x, k, p)
                                    .max(axis=2)) + numpy.log(weights))
            llv_sample = scipy.misc.logsumexp(llv_sample_subclone, axis=1)
            llv = llv_sample.sum()
            if llv > old_llv and llv < old_llv + 0.001:
                _LOGGER.info("EM is fully converged.")
                break
        else:
            _LOGGER.warning("Maximum iteration reached but the EM is not "
                            "fully converged.")
        return weights


class State2SubcloneEstimator(BaseDensityBasedSubcloneEstimator):
    """A density-based subclone estimator using 2-state-CNA SNVs."""

    def _infer(self):
        _LOGGER.info("Subclone estimated using SNVs with 2-state CNA.")
        merged = self._match_snv_to_isa_cna()
        if self.snvs.is_female:
            check = (self._snv_predictor.predictions &
                     ~merged["frac1"].isnull())
        else:
            check = (self._snv_predictor.predictions &
                     (merged["chromosome_x"] != "X") &
                     (merged["chromosome_y"] != "Y") &
                     ~merged["frac1"].isnull())
        samples = merged[check]
        vaf = self.snvs.sample_data.loc[samples.index, ("tumour", "vaf")]
        #For samples with CNV, there will always be cellularity.
        cellularity = self._cellularity_estimator.cellularity
        base_ratios = vaf * ((samples["major1"] + samples["minor1"]) *
                             (cellularity * samples["frac1"]) +
                             2 * (1 - cellularity * samples["frac1"]))
        diff1 = pandas.Series(0.0, index=vaf.index)
        diff2 = (samples["major1"] - 1) * samples["frac1"] * cellularity
        diff3 = (samples["minor1"] - 1) * samples["frac1"] * cellularity
        diff = pandas.concat([diff1, diff2, diff3], axis=1)
        na_check = base_ratios.isnull() | diff.isnull().any(axis=1)
        samples = samples[~na_check]
        base_ratios = base_ratios[~na_check]
        diff = diff[~na_check]
        ratios = base_ratios.values[:, None] - diff.values
        peaks = self._get_density_peaks(ratios.flatten())
        peaks = self._adjust_peaks_using_cellularity(peaks)
        self._update_subclones(samples, peaks)

    def _estimate_weights(self, samples, peaks):
        _LOGGER.debug("Found subclone prevalences: {}", peaks)
        peaks = peaks[None, :, None]
        cellularity = peaks.max()
        weights = numpy.repeat(1 / peaks.size,peaks.size).astype("f4")
        major = samples["major1"].values[:, None, None]
        minor = samples["minor1"].values[:, None, None]
        frac = samples["frac1"].values[:, None, None]
        denom = (frac * cellularity * (major + minor) +
                 (1 - frac * cellularity) * 2)
        numer = numpy.dstack((
            numpy.where(peaks < frac, 0, (major * frac * cellularity +
                                          peaks - frac * cellularity)),
            numpy.where(peaks < frac, 0, (minor * frac * cellularity +
                                          peaks - frac * cellularity)),
            numpy.repeat(peaks, len(samples), axis=0),
        ))
        p = numer / denom
        tumour_data = self.snvs.sample_data.loc[samples.index, "tumour"]
        x = tumour_data["alternative"].values[:, None, None]
        k = x + tumour_data["reference"].values[:, None, None]
        llv_sample_subclone = ((scipy.stats.binom.logpmf(x, k, p)
                                .max(axis=2)) + numpy.log(weights))
        llv_sample = scipy.misc.logsumexp(llv_sample_subclone, axis=1)
        llv = llv_sample.sum()
        for t in range(1000):
            old_llv = llv
            p_member = numpy.exp(llv_sample_subclone - llv_sample[:, None])
            weights = p_member.mean(axis=0)
            weights /= weights.sum()
            llv_sample_subclone = ((scipy.stats.binom.logpmf(x, k, p)
                                    .max(axis=2)) + numpy.log(weights))
            llv_sample = scipy.misc.logsumexp(llv_sample_subclone, axis=1)
            llv = llv_sample.sum()
            if llv > old_llv and llv < old_llv + 0.001:
                _LOGGER.info("EM is fully converged.")
                break
        else:
            _LOGGER.warning("Maximum iteration reached but the EM is not "
                            "fully converged.")
        return weights


class CombinedSubcloneEstimator(BaseDensityBasedSubcloneEstimator):
    """A subclone estimator combining basic estimators above."""

    def _infer(self):
        merged = self._match_snv_to_isa_cna()
        check = (self._snv_predictor.predictions &
                 (merged["chromosome_x"] != "X") &
                 (merged["chromosome_y"] != "Y") &
                 (merged["state_count"] < 1) & ~merged["frac1"].isnull())
        if check.sum() > 50:
            worker = State0SubcloneEstimator(
                self._snv_predictor, self._cellularity_estimator, self._snvs,
                self._cnas, self._precalc, self._exclude)
            self._subclones = worker.subclones
            return
        check = (self._snv_predictor.predictions &
                 (merged["chromosome_x"] != "X") &
                 (merged["chromosome_y"] != "Y") &
                 (merged["state_count"] < 2) & ~merged["frac1"].isnull())
        if check.sum() > 50:
            worker = State1SubcloneEstimator(
                self._snv_predictor, self._cellularity_estimator, self._snvs,
                self._cnas, self._precalc, self._exclude)
            self._subclones = worker.subclones
            return
        worker = State2SubcloneEstimator(
            self._snv_predictor, self._cellularity_estimator, self._snvs,
            self._cnas, self._precalc, self._exclude)
        self._subclones = worker.subclones

    def _estimate_weights(self, samples, peaks):
        pass


@six.add_metaclass(abc.ABCMeta)
class BasePhylogenyEstimator(BaseSolverComponent):
    """A phylogeny inference interface."""

    def __init__(self, subclone_estimator, *args, **kargs):
        """Create a BasePhylogenyEstimator object.

        Parameters
        ----------
        subclone_estimator : licl.BaseSubcloneEstimator
            A subclone estimator.
        snvs : licl.SNVCollection
            All SNV records for inference.
        cnas : licl.CNACollection
            All CNA records for inference.
        precalc: pathlib.Path or str-like, default=None
            A pre-calculated file that summarises average ploidy and
            tumour purity.
        exclude : list[str], default=[]
            Data that should be excluded during training.
        """
        super(BasePhylogenyEstimator, self).__init__(*args, **kargs)
        self._subclone_estimator = subclone_estimator
        self._phylogeny = None

    @property
    def phylogeny(self):
        """numpy.ndarray : A list of subclonal parents."""
        if self._phylogeny is None:
            self._infer()
        return self._phylogeny

    @abc.abstractmethod
    def _infer(self):
        pass


class LinearPhylogenyEstimator(BasePhylogenyEstimator):
    """A linear phylogeny estimator."""

    def _infer(self):
        _LOGGER.info("Phylogeny inference model: Linear phylogeny")
        parents = numpy.arange(len(self._subclone_estimator.subclones.columns))
        _LOGGER.debug("Phylogeny: {}", parents)
        self._phylogeny = parents - 1


class BetaPhylogenyEstimator(BasePhylogenyEstimator):
    """A phylogeny estimator assuming beta-distributed prevalences."""

    def _infer(self):
        _LOGGER.info("Phylogeny inference model: Beta distribution")
        parents = numpy.zeros(len(self._subclone_estimator.subclones.columns),
                              dtype="i1")
        parents[0] = -1
        if parents.size < 3:
            _LOGGER.debug("Phylogeny: {}", parents + 1)
            self._phylogeny = parents
            return
        fractions = numpy.zeros(parents.shape, dtype="f4")
        leftover = numpy.asarray(self._subclone_estimator.subclones.columns)
        fractions[0] = leftover[0]
        fractions[1] = leftover[1] / leftover[0]
        leftover[0] -= leftover[1]
        parents[2] = self._assign_2nd_subclone(leftover, fractions)
        for j in range(3, parents.size):
            parents[j] = self._assign(j, leftover, fractions)
        _LOGGER.debug("Phylogeny: {}", parents + 1)
        self._phylogeny = parents

    def _assign_2nd_subclone(self, leftover, fractions):
        if leftover[0] > leftover[2] + 0.1:
            fractions[2] = leftover[2] / leftover[0]
            leftover[0] -= leftover[2]
            return 0
        else:
            fractions[2] = leftover[2] / leftover[1]
            leftover[1] -= leftover[2]
            return 1

    def _assign(self, idx, leftover, fractions):
        check = leftover[idx] < leftover[:idx]
        new_frac = leftover[idx] / leftover[check]
        fracs = numpy.hstack((
            numpy.repeat(fractions[:idx, None], new_frac.size, axis=1),
            new_frac,
        ))
        likelihood = numpy.asarray([
            scipy.stats.beta(
                *scipy.stats.beta.fit(fracs[:, j], f0=1, floc=0, fscale=1)
            ).logpdf(fracs[:, j]).sum()
            for j in range(new_frac.size)
        ])
        parent = numpy.where(check)[0][likelihood.argmax()]
        fractions[idx] = leftover[idx] / leftover[parent]
        leftover[parent] -= leftover[idx]
        return parent


class Solver(object):
    """The default solver."""

    def __init__(self, false_snv_predictor_class=CombinedSNVPredictor,
                 cellularity_estimator_class=CombinedCellularityEstimator,
                 subclone_estimator_class=CombinedSubcloneEstimator,
                 phylogeny_estimator_class=BetaPhylogenyEstimator):
        """Create a Solver object.

        Parameters
        ----------
        false_snv_predictor_class : class, default=CombinedSNVPredictor
            The false SNV predictor class
        cellularity_estimator_class : class,
                default=CombinedCellularityEstimator
            The cellularity estimator class
        subclone_estimator_class : class
            The subclone estimator class
        phylogeny_estimator_class : class
            The phylogeny estimator class
        """
        self.false_snv_predictor_class = false_snv_predictor_class
        self.cellularity_estimator_class = cellularity_estimator_class
        self.subclone_estimator_class = subclone_estimator_class
        self.phylogeny_estimator_class = phylogeny_estimator_class

    def infer(self, *args, **kargs):
        """Infer the tumour subclone and phylogeny.

        Parameters
        ----------
        snvs : licl.SNVCollection
            All SNV records for inference.
        cnas : licl.CNACollection
            All CNA records for inference.
        precalc: pathlib.Path or str-like, default=None
            A pre-calculated file that summarises average ploidy and
            tumour purity.
        exclude : list[str], default=[]
            Data that should be excluded during training.
        """
        _LOGGER.info("Filtering SNVs for the fallback solution...")
        snv_pred = self.false_snv_predictor_class(*args, **kargs)
        cellularity = RandomGuessCellularityEstimator(*args, **kargs)
        cluster = SingleSubcloneEstimator(snv_pred, cellularity,
                                          *args, **kargs)
        phylogeny = LinearPhylogenyEstimator(cluster, *args, **kargs)
        self._report(snv_pred, cluster, phylogeny, matrix_score=False)
        _LOGGER.notice(
            "Current solution: a fallback solution with filtered SNVs.")
        _LOGGER.info("Estimating cellularity for the fallback solution...")
        cellularity = self.cellularity_estimator_class(*args, **kargs)
        cluster = SingleSubcloneEstimator(snv_pred, cellularity,
                                          *args, **kargs)
        phylogeny = LinearPhylogenyEstimator(cluster, *args, **kargs)
        self._report(snv_pred, cluster, phylogeny, matrix_score=False)
        _LOGGER.notice(
            "Current solution: a fallback solution with fixed cellularity.")
        _LOGGER.info("Estimating subclones...")
        cluster = self.subclone_estimator_class(snv_pred, cellularity,
                                                *args, **kargs)
        phylogeny = self.phylogeny_estimator_class(cluster, *args, **kargs)
        _LOGGER.info("Generating the final solution...")
        self._report(snv_pred, cluster, phylogeny)
        _LOGGER.notice("Current solution: Final estimated results.")
        _LOGGER.notice("Inference succeeded.")

    def _report(self, snv_pred, cluster, phylogeny, matrix_score=True):
        subclones = cluster.subclones
        with open("cellularity.txt", "w") as output:
            print("{:f}".format(subclones.columns.max()), file=output)
        with open("subclone_count.txt", "w") as output:
            print(len(subclones.columns), file=output)
        self._report_prevalence_table(snv_pred, cluster)
        numpy.savetxt("assignments.txt",
                      subclones.values.argmax(axis=1) + 1, "%d")
        phylogeny_list = numpy.vstack((
            numpy.arange(len(subclones.columns)),
            phylogeny.phylogeny,
        ))
        numpy.savetxt("phylogeny.txt", (phylogeny_list.T + 1), "%d", "\t")
        if matrix_score:
            self._report_matrices(cluster, phylogeny)

    def _report_prevalence_table(self, snv_pred, cluster):
        cluster_count = len(cluster.subclones.columns)
        assignments = numpy.where(snv_pred.predictions.values,
                                  cluster.subclones.values.argmax(axis=1),
                                  cluster_count)
        indices = numpy.arange(cluster_count + 1)
        snv_count = (assignments == indices[:, None]).sum(axis=1).astype("i4")
        table = pandas.DataFrame({
            "snv_count": snv_count,
        })
        prevalences = numpy.zeros(snv_count.shape, dtype="f4")
        prevalences[:cluster_count] = cluster.subclones.columns
        table.insert(1, "prevalences", prevalences)
        table.index += 1
        table.to_csv("prevalences.txt", sep="\t", float_format="%f",
                     header=False)

    def _report_matrices(self, cluster, phylogeny):
        snv_count, subclone_count = cluster.subclones.shape
        pcm = numpy.zeros((subclone_count, subclone_count))
        for idx, parent in enumerate(phylogeny.phylogeny):
            while parent != -1:
                pcm[parent, idx] = 1
                parent = phylogeny.phylogeny[parent]
        with open("assignment_score.txt", "wb") as ccm_file, \
             open("phylogeny_score.txt", "wb") as ad_file:
            for j in range(snv_count):
                prob_row = cluster.subclones.values[j, :][None, :]
                ccm_row = prob_row.dot(cluster.subclones.values.T)
                ccm_row = numpy.around(ccm_row, 6)
                ccm_row = numpy.abs(ccm_row.clip(min=0.0, max=1.0))
                ccm_row[0, j] = 1.0
                ad_row = prob_row.dot(pcm).dot(cluster.subclones.values.T)
                ad_row = numpy.around(ad_row, 6)
                ad_row = numpy.abs(ad_row.clip(min=0.0, max=1.0))
                ad_row[0, j] = 0.0
                ad_col = cluster.subclones.values.dot(pcm).dot(prob_row.T)
                ad_col = numpy.around(ad_col, 6)
                ad_col = numpy.abs(ad_col.clip(min=0.0, max=1.0))
                ad_col[j, 0] = 0.0
                diff = (ccm_row + ad_row + ad_col.T - 1 + 1e-6).clip(min=0.0)
                ad_row = numpy.around(ad_row - diff, 6)
                ad_row = numpy.abs(ad_row.clip(min=0.0, max=1.0))
                numpy.savetxt(ccm_file, ccm_row, "%f", "\t")
                numpy.savetxt(ad_file, ad_row, "%f", "\t")


def main():
    logging_handler = logbook.StreamHandler(sys.stdout, level="INFO")
    logging_handler.format_string = _LOG_FORMAT
    logging_handler.push_application()
    opts = _generate_argument_parser().parse_args()
    _LOGGER.info("Generating a simplistic fallback solution...")
    _output_initial_fallback(opts.vcf_input)
    _LOGGER.notice("Current solution: a single-cluster fallback.")
    try:
        vcf = SNVCollection.from_vcf(opts.vcf_input)
    except:
        _LOGGER.error("Fail to parse VCF file.")
        raise
    try:
        bat = CNACollection.from_battenberg(opts.cna_input)
    except:
        _LOGGER.error("Fail to parse Battenberg file.")
        raise
    solver = Solver()
    solver.infer(vcf, bat, opts.cellularity, opts.exclude)
    logging_handler.pop_application()
    return 0


def _generate_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "vcf_input",
        help="specify a vcf file for simple nucleotide variations",
        metavar="VCF",
    )
    parser.add_argument(
        "cna_input",
        help="specify a Battenberg file for copy number variations",
        metavar="CNA",
    )
    parser.add_argument(
        "--exclude-training-data", "-e",
        action="append",
        dest="exclude",
        help="exclude the specified training data from the model",
        metavar="DATA",
        default=[],
    )
    parser.add_argument(
        "--cellularity", "-c",
        dest="cellularity",
        help="specify a file to provide pre-calculated cellularity",
        metavar="CELLULARITY",
    )
    return parser


def _output_initial_fallback(snv_path):
    snv_count = 0
    with open(snv_path) as snvs, \
         open("assignments.txt", "wt") as output:
        for line in snvs:
            if line.startswith("#") or line.strip() == "":
                continue
            print(1, file=output)
            snv_count += 1
    with open("cellularity.txt", "wt") as output:
        print(_DEFAULT_CELLULARITY, file=output)
    with open("subclone_count.txt", "wt") as output:
        print(1, file=output)
    with open("prevalences.txt", "wt") as output:
        print(1, snv_count, _DEFAULT_CELLULARITY, file=output, sep="\t")
    with open("phylogeny.txt", "wt") as output:
        print(1, 0, file=output, sep="\t")
    ccm_row = numpy.ones((1, snv_count), dtype="f4")
    ad_row = numpy.zeros((1, snv_count), dtype="f4")
    with open("assignment_score.txt", "wb") as ccm_file, \
         open("phylogeny_score.txt", "wb") as ad_file:
        for j in range(snv_count):
            numpy.savetxt(ccm_file, ccm_row, "%f", "\t")
            numpy.savetxt(ad_file, ad_row, "%f", "\t")
