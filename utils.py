import torch, functools, inspect

def _typechk(obj, types):
    if not isinstance(types, tuple):
        types = (types,) 

    if not any(isinstance(obj, t) for t in types):
        fmt = "Expected {}, got {}"
        msg = fmt.format(types, type(obj))
        raise ValueError(msg)

def tchk(t, shape=None, dtype=None):
    _typechk(t, torch.Tensor)

    def shape_matches(actual, expected):
        n_ellipsis = expected.count(...)
        if n_ellipsis > 1:
            raise ValueError("Only one ellipsis can be used to specify `shape`")

        if len(actual) != len(expected) and n_ellipsis == 0:
            # no ellipsis, dimensionality mismatch
            return False

        # check if dimensions match, one by one
        for a_s, e_s in zip(actual, expected):
            if e_s != a_s:
                if e_s == -1:
                    # wildcard dimension, continue
                    continue
                elif e_s == ...:
                    # ellipsis - done checking from the front, skip to checking in reverse
                    break
                else:
                    # mismatch
                    return False

        if n_ellipsis == 0:
            # no ellipsis - we don't have to go in reverse
            return True

        # check in reverse
        for a_s, e_s in zip(actual[::-1], expected[::-1]):
            if e_s != a_s:
                if e_s == -1:
                    # wildcard dimension, continue
                    continue
                elif e_s == ...:
                    # arrived at the ellipsis - done checking
                    return True
                else:
                    # mismatch
                    return False

        raise AssertionError("Arrived at the end of procedure")

    if shape is not None and not shape_matches(t.shape, shape):
        fmt = "Expected shape {}, got {}"
        msg = fmt.format(shape, t.shape)
        raise ValueError(msg)

    if dtype is not None and not isinstance(dtype, tuple):
        dtype = (dtype, )

    if dtype is not None and not any (dt == t.dtype for dt in dtype):
        fmt = "Expected dtype {}, got {}"
        msg = fmt.format(dtype, t.dtype)
        raise ValueError(msg)

class ShapeDict(dict):
    def __init__(self):
        super(ShapeDict, self).__init__()

    def update(self, other):
        for key in other.keys():
            if key in self and not self[key] == other[key]:
                raise LabeledShapeError(key, self[key], other[key])
            else:
                self[key] = other[key]
                
class ShapeError(Exception):
    pass

class SizeMismatch(ShapeError):
    def __init__(self, dim, expected, found):
        self.dim = dim
        self.expected = expected
        self.found = found
        self.param_name = None

    def __str__(self):
        fmt = "Size mismatch on dimension {} of argument `{}` (found {}, expected {})"
        msg = fmt.format(self.dim, self.param_name, self.found, self.expected)
        return msg

class LabeledShapeError(ShapeError):
    def __init__(self, label, prev_val, new_val):
        self.label = label
        self.prev_val = prev_val
        self.new_val = new_val
        self.param_name = None

    def __str__(self):
        fmt = ("Label `{}` already had dimension {} bound to it, "
               "but it appears with dimension {} in tensor {}")
        msg = fmt.format(self.label, self.prev_val, self.new_val, self.param_name)
        return msg

def get_bindings(tensor, annotation):
    n_ellipsis = annotation.count(...)
    if n_ellipsis > 1:
        # TODO: check this condition earlier
        raise ValueError("Only one ellipsis can be used per annotation")

    if len(annotation) != len(tensor.shape) and n_ellipsis == 0:
        # no ellipsis, dimensionality mismatch
        fmt = "Annotation {} differs in size from tensor shape {} ({} vs {})"
        msg = fmt.format(annotation, tensor.shape, len(annotation), len(tensor.shape))
        raise ShapeError(msg)

    bindings = ShapeDict()
    # check if dimensions match, one by one
    for i, (dim, anno) in enumerate(zip(tensor.shape, annotation)):
        if isinstance(anno, str):
            # named wildcard, add to dict
            bindings.update({anno: dim})
        elif anno == ...:
            # ellipsis - done checking from the front, skip to checking in reverse
            break
        elif isinstance(anno, int) and anno != dim:
            if anno == -1:
                # anonymous wildcard dimension, continue
                continue
            else:
                raise SizeMismatch(i, anno, dim)

    if n_ellipsis == 0:
        # no ellipsis - we don't have to go in reverse
        return bindings

    # there was an ellipsis, we have to check in reverse
    for i, (dim, anno) in enumerate(zip(tensor.shape[::-1], annotation[::-1])):
        if isinstance(anno, str):
            # named wildcard, add to dict
            bindings.update({anno: dim})
        elif anno == ...:
            # ellipsis - done checking from the back, return
            return bindings
        elif isinstance(anno, int) and anno != dim:
            if anno == -1:
                # anonymous wildcard dimension, continue
                continue
            else:
                raise SizeMismatch(len(annotation) - i, anno, dim)

    raise AssertionError("Arrived at the end of procedure")

def shape_checked(func):
    sig = inspect.signature(func)

    checked_parameters = dict()
    for i, parameter in enumerate(sig.parameters.values()):
        if isinstance(parameter.annotation, list):
            checked_parameters[i] = parameter

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        # check input
        shape_bindings = ShapeDict()
        for i, arg in enumerate(args):
            if i in checked_parameters:
                param = checked_parameters[i]
                try:
                    shapes = get_bindings(arg, param.annotation)
                    shape_bindings.update(shapes)
                except ShapeError as e:
                    e.param_name = param.name
                    raise

        result = func(*args, **kwargs)

        if isinstance(sig.return_annotation, list):
            try:
                shapes = get_bindings(result, sig.return_annotation)
                shape_bindings.update(shapes)
            except ShapeError as e:
                e.param_name = '<return value>'
                raise

        return result

    return wrapped

if __name__ == '__main__':
    import unittest
    
    class ShapeCheckedTests(unittest.TestCase):
        def test_wrap_no_anno(self):
            def f(t1, t2): # t1: [3, 5], t2: [5, 3] -> [3]
                return (t1.transpose(0, 1) * t2).sum(dim=0)
                 
            t1 = torch.randn(3, 5)
            t2 = torch.randn(5, 3)

            self.assertTrue((f(t1, t2) == shape_checked(f)(t1, t2)).all())

        def test_wrap_correct(self):
            def f(t1: [3, 5], t2: [5, 3]) -> [3]:
                return (t1.transpose(0, 1) * t2).sum(dim=0)
                 
            t1 = torch.randn(3, 5)
            t2 = torch.randn(5, 3)

            self.assertTrue((f(t1, t2) == shape_checked(f)(t1, t2)).all())

        def test_fails_wrong_parameter(self):
            def f(t1: [3, 3], t2: [5, 3]) -> [3]:
                return (t1.transpose(0, 1) * t2).sum(dim=0)
                 
            t1 = torch.randn(3, 5)
            t2 = torch.randn(5, 3)

            msg = "Size mismatch on dimension 1 of argument `t1` (found 5, expected 3)"
            with self.assertRaises(ShapeError) as ex:
                shape_checked(f)(t1, t2)
            self.assertEqual(str(ex.exception), msg)

        def test_fails_wrong_return(self):
            def f(t1: [3, 5], t2: [5, 3]) -> [5]:
                return (t1.transpose(0, 1) * t2).sum(dim=0)
                 
            t1 = torch.randn(3, 5)
            t2 = torch.randn(5, 3)

            msg = ("Size mismatch on dimension 0 of argument "
                   "`<return value>` (found 3, expected 5)")
            with self.assertRaises(ShapeError) as ex:
                shape_checked(f)(t1, t2)
            self.assertEqual(str(ex.exception), msg)

        def test_fails_parameter_label_mismatch(self):
            def f(t1: [3, 'a'], t2: ['a', 3]) -> [3]:
                return (t1.transpose(0, 1) * t2).sum(dim=0)
                 
            t1 = torch.randn(3, 4)
            t2 = torch.randn(5, 3)

            with self.assertRaises(ShapeError):
                shape_checked(f)(t1, t2)

        def test_fails_return_label_mismatch(self):
            def f(t1: [5, 'a'], t2: ['a', 5]) -> ['a']:
                return (t1.transpose(0, 1) * t2).sum(dim=0)
                 
            t1 = torch.randn(3, 5)
            t2 = torch.randn(5, 3)

            with self.assertRaises(ShapeError):
                shape_checked(f)(t1, t2)

        def test_succeeds_ellipsis(self):
            def f(t1: [5, ..., 'a'], t2: ['a', ..., 5]) -> ['a']:
                return (t1.transpose(0, 3) * t2).sum(dim=(1, 2, 3))
                 
            t1 = torch.randn(5, 1, 2, 3)
            t2 = torch.randn(3, 1, 2, 5)

            self.assertTrue((f(t1, t2) == shape_checked(f)(t1, t2)).all())

    unittest.main(failfast=True)
