using System;
using JetBrains.Annotations;
using MathNet.Numerics.LinearAlgebra.Double;

namespace DirichletProblem
{

    public class DirichletProblemSolver
    {
        [NotNull]
        private readonly SparseMatrix _innerPointsMatrix;

        private readonly Matrix _matrix;

        private readonly Vector _rightPartVector;

        private const double A = 1.0;
        private const double B = 1.2;
        private static double _h;

        private static double _maxEigenValue;
        private static bool _maxEigenValueFound = false;

        public double MaxEigenValue
        {
            get
            {
                if (_maxEigenValueFound)
                    return _maxEigenValue;
                else
                {
                    Vector z0 = new DenseVector(_matrix.RowCount);
                    z0[0] = 1;
                    var approximation1 = 0.0;
                    var approximation2 = 0.0;

                    var z1 = _matrix * z0;
                    approximation2 = z1[0] / z0[0];
                    approximation1 = approximation2 + 1;

                    while (Math.Abs(approximation2 - approximation1) > 1.0e-5)
                    {
                        approximation1 = approximation2;
                        z1.CopyTo(z0);

                        z1 = _matrix * z0;
                        approximation2 = z1[0] / z0[0];

                        z1 = z1.Normalize(2);
                    }

                    _maxEigenValueFound = true;
                    _maxEigenValue = approximation2;

                    return approximation2;
                }
            }
        }

        private static double _minEigenValue;
        private static bool _minEigenValueFound = false;

        public double MinEigenValue
        {
            get
            {
                if (_minEigenValueFound)
                    return _minEigenValue;
                else
                {
                    Vector z0 = new DenseVector(_matrix.RowCount);
                    z0[0] = 1;
                    var approximation1 = 0.0;
                    var approximation2 = 0.0;

                    var z1 = (_maxEigenValue * DenseMatrix.CreateIdentity(_matrix.RowCount) - _matrix) * z0;
                    approximation2 = z1.DotProduct(z0) / z0.DotProduct(z0);
                    approximation1 = approximation2 + 1;

                    while (Math.Abs(approximation2 - approximation1) > 1.0e-5)
                    {
                        approximation1 = approximation2;
                        z1.CopyTo(z0);

                        z1 = (_maxEigenValue * DenseMatrix.CreateIdentity(_matrix.RowCount) - _matrix) * z0;
                        approximation2 = z1.DotProduct(z0) / z0.DotProduct(z0);

                        z1 = z1.Normalize(2);
                    }

                    _minEigenValueFound = true;
                    _minEigenValue = approximation2;

                    return approximation2;
                }
            }
        }

        static double F(double x, double y)
        {
            return 0.2 * Math.Exp(x) * Math.Cos(y);
        }

        static double G(double x, double y)
        {
            return Math.Exp(x) * Math.Cos(y);
        }


        public DirichletProblemSolver(int partitionSize)
        {
            // 1. Construct points matrix, then construct a vector from points matrix
            // 2. Form matrix, whilst calculating right part of the equation

            // Setting enumeration

            var counter = 1;
            _h = 1.0 / partitionSize;

            _innerPointsMatrix = new SparseMatrix(partitionSize + 1);
            for (var i = 1; i < (_innerPointsMatrix.RowCount / 2); i++)
            {
                for (var j = 1; j < _innerPointsMatrix.ColumnCount - 1; j++)
                {
                    _innerPointsMatrix[i, j] = counter++;
                }
            }
            for (var i = _innerPointsMatrix.RowCount / 2; i < _innerPointsMatrix.RowCount; i++)
            {
                for (var j = 1; j < _innerPointsMatrix.ColumnCount / 2 - i + _innerPointsMatrix.RowCount / 2; j++)
                {
                    _innerPointsMatrix[i, j] = counter++;
                }
            }

            // End of setting enumeration

            var matrixSize = (partitionSize - 1) * (partitionSize / 2 - 1) +
                             (partitionSize / 2 - 1) * (partitionSize / 2) / 2;

            _rightPartVector = new DenseVector(matrixSize);

            _matrix = new SparseMatrix(matrixSize);

            for (var k = 0; k < matrixSize; k++)
            {
                for (var i = 0; i < _innerPointsMatrix.RowCount; i++)
                {
                    for (var j = 0; j < _innerPointsMatrix.ColumnCount; j++)
                    {
                        if (!(Math.Abs(_innerPointsMatrix[i, j] - (k + 1)) < 0.1)) continue;
                        _rightPartVector[k] = F(i * _h, j * _h) * _h * _h;
                        _matrix[k, k] = 2 * A + 2 * B;

                        if (Math.Abs(_innerPointsMatrix[i - 1, j]) < 0.1)
                        {
                            _rightPartVector[k] += G(i * _h - _h, j * _h) * A;
                        }
                        else
                        {
                            _matrix[k, Convert.ToInt32(_innerPointsMatrix[i - 1, j]) - 1] = -A;
                        }

                        if (Math.Abs(_innerPointsMatrix[i + 1, j]) < 0.1)
                        {
                            _rightPartVector[k] += G(i * _h + _h, j * _h) * A;
                        }
                        else
                        {
                            _matrix[k, Convert.ToInt32(_innerPointsMatrix[i + 1, j]) - 1] = -A;
                        }

                        if (Math.Abs(_innerPointsMatrix[i, j + 1]) < 0.1)
                        {
                            _rightPartVector[k] += G(i * _h, j * _h + _h) * B;
                        }
                        else
                        {
                            _matrix[k, Convert.ToInt32(_innerPointsMatrix[i, j + 1]) - 1] = -B;
                        }

                        if (Math.Abs(_innerPointsMatrix[i, j - 1]) < 0.1)
                        {
                            _rightPartVector[k] += G(i * _h, j * _h - _h) * B;
                        }
                        else
                        {
                            _matrix[k, Convert.ToInt32(_innerPointsMatrix[i, j - 1]) - 1] = -B;
                        }
                    }
                }
            }
        }

        public void Show(Matrix matrix)
        {
            for (var i = 0; i < matrix.RowCount; i++)
            {
                for (var j = 0; j < matrix.ColumnCount; j++)
                {
                    Console.Write(matrix[i, j]);
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }

        public Vector SolveJacobi()
        {
            const double epsilon = 1.0e-5;

            var x0 = new DenseVector(_matrix.RowCount);
            x0[0] = 1;

            while ((_matrix * x0 - _rightPartVector).Norm(2.0) >= epsilon)
            {
                var vect = _matrix * x0;

                for (var i = 0; i < _matrix.RowCount; i++)
                {
                    vect[i] /= _matrix[i, i];
                }
                x0 = (DenseVector)x0.Subtract(vect) + (DenseVector)_rightPartVector.Divide(_matrix[0, 0]);
            }

            return x0;
        }

        public double[] SolveJacobiDoubles()
        {
            return SolveJacobi().ToArray();
        }

        public bool JacobiIteration(Vector vector, double epsilon)
        {
            if (vector == null)
            {
                vector = new DenseVector(_matrix.RowCount);
            }

            var vect = _matrix * vector;

            for (var i = 0; i < _matrix.RowCount; i++)
            {
                vect[i] /= _matrix[i, i];
            }
            vector = (DenseVector)vector.Subtract(vect) + (DenseVector)_rightPartVector.Divide(_matrix[0, 0]);

            return (_matrix * vector - _rightPartVector).Norm(2) < epsilon;
        }

        public bool JacobiIterationDoubles(ref double[] array, double epsilon)
        {
            var vector = new DenseVector(array);
            var info =  JacobiIteration(vector, epsilon);
            array = vector.ToArray();
            return info;
        }

        public void GetCoordinate(int number, out double x, out double y)
        {
            x = -1;
            y = -1;
            for (var i = 0; i < _innerPointsMatrix.RowCount; i++)
            {
                for (var j = 0; j < _innerPointsMatrix.ColumnCount; j++)
                {
                    if (!(Math.Abs(_innerPointsMatrix[i, j] - number) < 0.1)) continue;
                    y = i * _h;
                    x = j * _h;
                }
            }
        }

        public void CreateCoordsFile(string path)
        {
            var fileStream = new System.IO.StreamWriter(path);

            double x, y;

            var v = SolveJacobi();

            Show(_innerPointsMatrix);

            fileStream.Write("{");

            for (int i = 0; i < _matrix.RowCount; i++)
            {
                GetCoordinate(i + 1, out x, out y);
                fileStream.Write("{" + x + "," + y + "," + v[i] + "},");
            }

            fileStream.Write("}");
            fileStream.Close();

        }
    }

}
