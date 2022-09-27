import unittest
import transform
import numpy as np
class TestStringMethods(unittest.TestCase):
    def test_transformations(self):
        for tr in transform.Transformation:
            shape = (8,8)
            pts = np.asarray((np.linspace(-1+1/shape[0],1+1/shape[0],shape[0],endpoint=False), np.linspace(-1+1/shape[1],1+1/shape[1],shape[1],endpoint=False)))
            pts *= 0.5
            cart = transform.tex2cart(*pts,tr = tr)
            pts_res = transform.cart2tex(*cart,tr = tr)
            np.testing.assert_array_almost_equal(pts, pts_res,err_msg=tr.name)
            #,msg=tr.name + " Arrays not equal " + str(pts) + " " + str(pts_res)


    def test(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()
