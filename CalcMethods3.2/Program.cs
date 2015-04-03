using System;
using System.IO;
using DirichletProblem;
using MathNet.Numerics.LinearAlgebra.Double;

namespace CalcMethods3._2
{
    
    class Program
    {
        static void Main(string[] args)
        {
            var dps = new DirichletProblemSolver(32);
            //Console.WriteLine("Max: " + dps.MaxEigenValue);
            //Console.WriteLine("Min: " + dps.MinEigenValue);


            dps.CreateCoordsFile("D:\\data.txt");
        }
    }
}
