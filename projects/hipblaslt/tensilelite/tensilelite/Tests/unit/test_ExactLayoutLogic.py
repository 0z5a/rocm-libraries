# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Verify stride-sensitive exact routing survives canonical layout conversion."""
from copy import deepcopy
from importlib import import_module
from types import SimpleNamespace

import pytest

from tensilelite.Common import state
from tensilelite.Common.GlobalParameters import (
    assignGlobalParameters,
    globalParameters,
    restoreDefaultGlobalParameters,
)
from tensilelite.SolutionLibrary import (
    MasterSolutionLibrary,
    PlaceholderLibrary,
    PredicateLibrary,
)
from tensilelite.Properties import Predicate
from tensilelite.TensileCreateLibrary.ParseArguments import parseArguments

pytestmark = pytest.mark.unit


def test_stride_conflicting_zero_efficiency_routes_and_reindex():
    solutions = [SimpleNamespace(index=0), SimpleNamespace(index=1)]
    table = [[[64, 80, 1, 32, 104, 103, 101, 102], [0, 0.0]],
             [[64, 80, 1, 32, 204, 203, 201, 202], [1, 0.0]]]
    rows = MasterSolutionLibrary.exactLayoutRows(table, solutions)
    assert len(rows) == 2
    for row, expected in zip(rows, ((101, 102, 103, 104), (201, 202, 203, 204))):
        predicate = state(row['predicate'])
        assert [p['type'] for p in predicate['value']] == [
            'StrideAEqual', 'StrideBEqual', 'StrideCEqual', 'StrideDEqual']
        assert tuple(p['value'] for p in predicate['value']) == expected
        assert all(p['index'] == 1 for p in predicate['value'])
        match = row['library'].rows[0]['library']
        assert match.distance == 'Equality' and not match.useKdTree
        assert match.table[0]['key'] == [64, 80, 1, 32]
        assert match.table[0]['speed'] == 0.0
    solutions[0].index, solutions[1].index = 288135, 288136
    assert [state(row['library'])['rows'][0]['library']['table'][0]['index']
            for row in rows] == [288135, 288136]
    # PredicateLibrary merges sort rows; layout guards must precede fallback.
    library = PredicateLibrary(tag='Problem')
    library.rows = [{'predicate': Predicate(tag='GridBasedMatching'), 'library': rows[0]['library']}]
    incoming = PredicateLibrary(tag='Problem')
    incoming.rows = rows
    library.merge(incoming)
    assert state(library)['rows'][-1]['predicate']['type'] == 'GridBasedMatching'


@pytest.mark.parametrize('row', [
    [[64, 80, 1, 32, 104, 103, 101, 0], [0, 0.0]],
    [[64, 80, 1, 32, 104, 103, 101, True], [0, 0.0]],
    [[64, 80, 1, 32, 104, 103, 101, 102], [8, 0.0]],
    [[64, 80, 1, 32, 104, 103, 101, 102], [0, float('nan')]],
    [[64, 80, 1, 32, 104, 103, 101, 102], [0, -1.0]],
])
def test_invalid_layout_is_rejected(row):
    with pytest.raises(ValueError):
        MasterSolutionLibrary.exactLayoutRows([row], [SimpleNamespace(index=0)])


def test_duplicate_exact_layout_is_rejected():
    row = [[64, 80, 1, 32, 104, 103, 101, 102], [0, 0.0]]
    with pytest.raises(ValueError, match='Duplicate'):
        MasterSolutionLibrary.exactLayoutRows([row, row], [SimpleNamespace(index=0)])


@pytest.mark.parametrize('options, expected', [([], 0), (['--solution-index-start', '288135'], 288135)])
def test_index_start_reaches_generator_arguments(monkeypatch, options, expected):
    monkeypatch.setattr('sys.argv', ['TensileCreateLibrary', 'logic', 'output', 'HIP'] + options)
    assert parseArguments()['SolutionIndexStart'] == expected


@pytest.mark.parametrize('options, expected', [([], 0), (['--solution-index-start', '288135'], 288135)])
def test_index_start_survives_global_parameter_assignment(monkeypatch, options, expected):
    monkeypatch.setattr('sys.argv', ['TensileCreateLibrary', 'logic', 'output', 'HIP'] + options)
    arguments = parseArguments()
    try:
        assignGlobalParameters(arguments, {})
        assert arguments['SolutionIndexStart'] == expected
        assert 'SolutionIndexStart' not in globalParameters
    finally:
        restoreDefaultGlobalParameters()


@pytest.mark.parametrize('value', ['-1', '1.5', 'invalid'])
def test_invalid_index_start_is_rejected(monkeypatch, value):
    monkeypatch.setattr('sys.argv', ['TensileCreateLibrary', 'logic', 'output', 'HIP',
                                   '--solution-index-start', value])
    with pytest.raises(SystemExit):
        parseArguments()


_OMITTED = object()
_FLAG = 'ExactLogicIncludesLeadingDimensions'


class _OriginalSolution:
    def __init__(self, data):
        self._state = deepcopy(data)


class _FactorySolution:
    """Skip kernel derivation while retaining real library index references."""

    def __init__(self, data):
        self.index = data['SolutionIndex']
        self.srcName = data['SourceName']
        self.originalSolution = _OriginalSolution(data)

    @classmethod
    def FromSolutionStruct(cls, data, *args):
        return cls(data)

    def state(self):
        return {'SolutionIndex': self.index}


class _FactoryProblemType:
    operationIdentifier = 'test_HSS_NT'
    fusedGemmA2A = False

    def predicates(self, includeBatch=False, includeType=False):
        return []


def _logic_state(flag=True, include_layouts=True):
    # Equal dimensions with conflicting LDs expose accidental dimension-only
    # routing. A third solution gives the original fallback a distinct index.
    table = [[[48, 48, 1, 16], [2, 7.0]]]
    if include_layouts:
        table += [
            [[64, 80, 1, 32, 104, 103, 101, 102], [0, 0.0]],
            [[64, 80, 1, 32, 204, 203, 201, 202], [1, 0.0]],
        ]
    data = {
        'ArchitectureName': 'fallback',
        'CUCount': 0,
        'ProblemType': {'OperationType': 'GEMM', 'IndexAssignmentsLD': [4, 5, 6, 7]},
        'LibraryType': 'Matching',
        'Library': {'distance': 'GridBased', 'indexOrder': [2, 3, 0, 1],
                    'table': table, 'useKdTree': True},
    }
    if flag is not _OMITTED:
        data[_FLAG] = flag
    return data


def _factory(monkeypatch, data):
    monkeypatch.setattr(
        'tensilelite.Contractions.ProblemType.FromOriginalState',
        lambda problem: _FactoryProblemType(),
    )
    originals = [
        {'SolutionIndex': index, 'SourceName': name, 'codeObjectFile': 'tiny_hss.co'}
        for index, name in enumerate(('z-last.yaml', 'a-first.yaml', 'm-middle.yaml'))
    ]
    library, _ = MasterSolutionLibrary.FromOriginalState(
        data, originals, splitGSU=False, printSolutionRejectionReason=False,
        printIndexAssignmentInfo=False, assembler=None, isaInfoMap={},
        lazyLibraryLoading=False, solutionClass=_FactorySolution,
    )
    return library


def _state_nodes(value):
    """Walk serialized output without assuming the outer predicate levels."""
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _state_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _state_nodes(child)


def _selection_state(library):
    selections = [
        node for node in _state_nodes(state(library.library))
        if any(row['predicate']['type'] == 'GridBasedMatching'
               for row in node.get('rows', []))
    ]
    assert len(selections) == 1
    return selections[0]


def _assert_layout_selection(library, expected_indices, use_kdtree=True):
    selection = _selection_state(library)
    assert len(selection['rows']) == 3
    for row, strides, index in zip(selection['rows'][:2],
                                   ((101, 102, 103, 104), (201, 202, 203, 204)),
                                   expected_indices[:2]):
        assert row['predicate']['type'] == 'And'
        assert row['predicate']['value'] == [
            {'type': f'Stride{tensor}Equal', 'index': 1, 'value': stride}
            for tensor, stride in zip('ABCD', strides)
        ]
        assert row['library']['rows'][0]['predicate'] == {'type': 'EqualityMatching'}
        matching = row['library']['rows'][0]['library']
        assert matching['distance'] == 'Equality'
        assert matching['useKdTree'] is False
        assert matching['properties'] == [
            {'type': tag, 'index': 0}
            for tag in ('FreeSizeA', 'FreeSizeB', 'BatchSize', 'BoundSize')
        ]
        assert matching['table'] == [
            {'key': [64, 80, 1, 32], 'index': index, 'speed': 0.0}
        ]
    fallback = selection['rows'][-1]
    assert fallback['predicate'] == {'type': 'GridBasedMatching'}
    assert fallback['library']['distance'] == 'GridBased'
    assert fallback['library']['useKdTree'] is use_kdtree
    assert fallback['library']['table'] == [
        {'key': [48, 48, 1, 16], 'index': expected_indices[2]}
    ]


def test_factory_opt_in_routes_conflicting_layouts_and_retains_fallback(monkeypatch):
    data = _logic_state()
    original = deepcopy(data)
    library = _factory(monkeypatch, data)
    _assert_layout_selection(library, (0, 1, 2))
    assert data == original


@pytest.mark.parametrize('flag', [_OMITTED, False], ids=['omitted', 'false'])
@pytest.mark.parametrize('include_layouts', [False, True], ids=['four-field', 'mixed'])
def test_factory_without_opt_in_keeps_legacy_projection(monkeypatch, flag, include_layouts):
    data = _logic_state(flag, include_layouts)
    original = deepcopy(data)
    library = _factory(monkeypatch, data)
    selection = _selection_state(library)
    assert len(selection['rows']) == 1
    fallback = selection['rows'][0]
    assert fallback['predicate'] == {'type': 'GridBasedMatching'}
    assert fallback['library']['distance'] == 'GridBased'
    assert fallback['library']['useKdTree'] is True
    expected = [{'key': [48, 48, 1, 16], 'index': 2}]
    if include_layouts:
        expected += [{'key': [64, 80, 1, 32], 'index': 0},
                     {'key': [64, 80, 1, 32], 'index': 1}]
    assert fallback['library']['table'] == expected
    assert state(library) == state(_factory(monkeypatch, _logic_state(_OMITTED, include_layouts)))
    assert data == original


@pytest.mark.parametrize('flag', [None, 0, 1, 'true', [], {}])
def test_factory_rejects_non_boolean_opt_in(monkeypatch, flag):
    with pytest.raises(ValueError, match='Boolean'):
        _factory(monkeypatch, _logic_state(flag))


@pytest.mark.parametrize('include_layouts', [False, True], ids=['four-field', 'mixed'])
@pytest.mark.parametrize('section, key, value', [
    (None, 'LibraryType', 'FreeSize'),
    ('Library', 'distance', 'Equality'),
    ('Library', 'distance', 'Range'),
    ('ProblemType', 'OperationType', 'TensorContraction'),
    ('ProblemType', 'IndexAssignmentsLD', [0, 1, 2, 3]),
    ('Library', 'indexOrder', [0, 1, 2, 3]),
])
def test_factory_rejects_opt_in_with_unsupported_metadata(
        monkeypatch, include_layouts, section, key, value):
    data = _logic_state(include_layouts=include_layouts)
    target = data if section is None else data[section]
    target[key] = value
    with pytest.raises(ValueError):
        _factory(monkeypatch, data)


@pytest.mark.parametrize('row', [
    [[64, 80, 1, 32, 104, 103, 101, 0], [0, 0.0]],
    [[64, 80, 1, 32, 104, 103, 101, True], [0, 0.0]],
    [[64, 80, 1, 32, 104, 103, 101, 102], [8, 0.0]],
    [[64, 80, 1, 32, 104, 103, 101, 102], [0, float('nan')]],
    [[64, 80, 1, 32, 104, 103, 101, 102], [0, True]],
])
def test_factory_applies_layout_validation(monkeypatch, row):
    data = _logic_state()
    data['Library']['table'][1] = row
    with pytest.raises(ValueError):
        _factory(monkeypatch, data)


def test_factory_rejects_duplicate_opt_in_layout(monkeypatch):
    data = _logic_state()
    data['Library']['table'].append(deepcopy(data['Library']['table'][1]))
    with pytest.raises(ValueError, match='Duplicate'):
        _factory(monkeypatch, data)


def _generate(monkeypatch, lazy, start):
    generator = import_module('tensilelite.TensileCreateLibrary.Run')
    leaf = _factory(monkeypatch, _logic_state())
    if lazy:
        library = MasterSolutionLibrary({}, PlaceholderLibrary('tiny_hss'))
        library.lazyLibraries['tiny_hss'] = leaf
    else:
        library = leaf
    # Input parsing and compiler target discovery are outside these tests.
    # Keep the real allocation loop, lazy sorting, library objects and wrappers.
    monkeypatch.setattr(generator, 'ParallelMap2', lambda *args, **kwargs: iter([
        (None, 'gfx942', {}, None, None, library, {})
    ]))
    monkeypatch.setattr(generator, 'archRenamesOf', lambda archs, compiler: {})
    arguments = {'Architecture': 'gfx942', 'LazyLibraryLoading': lazy,
                 'CodeObjectVersion': '4', 'GenSolTable': False}
    if start is not _OMITTED:
        arguments['SolutionIndexStart'] = start
    solutions, masters, code_objects = generator.generateLogicDataAndSolutions(
        ['tiny_hss.yaml'], arguments, SimpleNamespace(path='unused'), {},
    )
    return list(solutions), masters, code_objects


@pytest.mark.parametrize('lazy', [False, True], ids=['non-lazy', 'lazy'])
@pytest.mark.parametrize('start', [_OMITTED, 0, 288135], ids=['omitted', 'zero', 'offset'])
def test_generator_allocates_indices_and_updates_layout_references(monkeypatch, lazy, start):
    solutions, masters, code_objects = _generate(monkeypatch, lazy, start)
    first = 0 if start is _OMITTED else start
    assert set(masters) == {'gfx942'}
    master = masters['gfx942']
    assert master.version == '4'
    if lazy:
        assert master.solutions == {}
        assert list(master.lazyLibraries) == ['tiny_hss']
        leaf = master.lazyLibraries['tiny_hss']
        # Lazy allocation orders source filenames: a, m, z, not local indices.
        assert [solution.srcName for solution in leaf.solutions.values()] == [
            'a-first.yaml', 'm-middle.yaml', 'z-last.yaml'
        ]
        expected_refs = (first + 2, first, first + 1)
        filename = 'tiny_hss'
    else:
        assert master.lazyLibraries == {}
        leaf = master
        expected_refs = (first, first + 1, first + 2)
        filename = 'tiny_hss.co'
    assert [solution.index for solution in leaf.solutions.values()] == list(range(first, first + 3))
    assert state(leaf)['solutions'] == [
        {'SolutionIndex': index} for index in range(first, first + 3)
    ]
    _assert_layout_selection(leaf, expected_refs)
    assert solutions == [solution.originalSolution for solution in leaf.solutions.values()]
    assert all(solution._state['codeObjectFile'] == filename for solution in solutions)
    assert code_objects == {first: filename}


@pytest.mark.parametrize('start', [-1, True, 1.5, '288135', None])
def test_generator_rejects_invalid_start_without_cli(monkeypatch, start):
    with pytest.raises(ValueError, match='nonnegative integer'):
        _generate(monkeypatch, False, start)


def _io_logic_state(flag, carrier='bare'):
    data = _logic_state(flag)
    data.update({
        'MinimumRequiredVersion': '4.0.0', 'ScheduleName': 'tiny_hss',
        'CUCount': 64 if carrier == 'cu' else None,
        'UseKdTree': carrier == 'kdtree', 'DeviceNames': ['fallback'],
        'Solutions': [], 'IndexOrder': data['Library']['indexOrder'],
        'ExactLogic': data['Library']['table'], 'RangeLogic': [],
        'LibraryType': 'GridBased',
    })
    return data


@pytest.mark.parametrize('flag', [_OMITTED, False, True], ids=['omitted', 'false', 'true'])
@pytest.mark.parametrize('carrier', ['bare', 'cu', 'kdtree'])
def test_layout_flag_roundtrip_preserves_factory_interpretation(monkeypatch, flag, carrier):
    library_io = import_module('tensilelite.LibraryIO')
    data = _io_logic_state(flag, carrier)
    original = deepcopy(data)
    raw = library_io.rawLibraryLogic(data)
    expected_architecture = {'Architecture': 'fallback'}
    if carrier == 'cu':
        expected_architecture['CUCount'] = 64
    if carrier == 'kdtree':
        expected_architecture['UseKdTree'] = True
    if flag is not _OMITTED:
        expected_architecture[_FLAG] = flag
    if carrier == 'bare' and flag is _OMITTED:
        expected_architecture = 'fallback'
    assert raw[2] == expected_architecture
    # rawLibraryLogic returns the optional list-format fields as one tail list.
    parsed = library_io.parseLibraryLogicList(list(raw[:9]) + raw[9])
    if flag is _OMITTED:
        assert _FLAG not in parsed
    else:
        assert parsed[_FLAG] is flag
    assert parsed['ExactLogic'] == data['ExactLogic']
    library = _factory(monkeypatch, parsed)
    if flag is True:
        _assert_layout_selection(library, (0, 1, 2), use_kdtree=carrier == 'kdtree')
    else:
        selection = _selection_state(library)
        assert len(selection['rows']) == 1
        assert selection['rows'][0]['library']['table'] == [
            {'key': [48, 48, 1, 16], 'index': 2},
            {'key': [64, 80, 1, 32], 'index': 0},
            {'key': [64, 80, 1, 32], 'index': 1},
        ]
    assert data == original


@pytest.mark.parametrize('flag', [None, 0, 1, 'true', [], {}])
@pytest.mark.parametrize('direction', ['dict-writer', 'list-reader'])
def test_logic_io_rejects_non_boolean_layout_flag(flag, direction):
    library_io = import_module('tensilelite.LibraryIO')
    if direction == 'dict-writer':
        with pytest.raises(ValueError, match='Boolean'):
            library_io.rawLibraryLogic(_io_logic_state(flag))
    else:
        raw = library_io.rawLibraryLogic(_io_logic_state(True))
        data = list(raw[:9]) + raw[9]
        data[2][_FLAG] = flag
        with pytest.raises(ValueError, match='Boolean'):
            library_io.parseLibraryLogicList(data)
