#ifndef SOLVER_H
#define SOLVER_H

#include "matrix.h"
#include "particle.h"
#include "lineareq.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

#define RELAXATION_PARAMETER 1.

class Solver
{
public:
    Solver();
    virtual ~Solver();

    //Eigen::SparseMatrix<double> m_invM_Eigen, m_JT_Eigen, m_A_Eigen;
    Eigen::MatrixXd m_invM_Eigen, m_JT_Eigen;
    SparseMatrix m_invM, m_JT, m_A;
    double *m_b, *m_gamma, *m_dp;
    Eigen::VectorXd m_b_Eigen, m_gamma_Eigen, m_dp_Eigen;
    int *m_counts;
    int m_nParts, m_nCons;
    LinearEquation m_eq;

    int getCount(int idx);

    void setupM(QList<Particle *> *particles, bool contact = false);
    void setupSizes(int numParts, QList<Constraint *> *constraints);
    void solveAndUpdate(QList<Particle *> *particles, QList<Constraint *> *constraints, bool stabile = false);
};

#endif // SOLVER_H
