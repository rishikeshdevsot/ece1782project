#include "solver.h"

Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

Solver::Solver()
{
    m_b = new double[2];
    m_gamma = new double[2];
    m_nCons = -1;

    m_dp = new double[2];
    m_counts = new int[2];
    m_nParts = -1;
}

Solver::~Solver()
{
    delete[] m_b;
    delete[] m_gamma;
    delete[] m_dp;
    delete[] m_counts;
}

int Solver::getCount(int idx)
{
    return m_counts[idx];
}

void Solver::setupM(QList<Particle *> *particles, bool contact)
{
    m_invM_Eigen.resize(particles->size()*2,particles->size()*2);
    m_invM_Eigen.Zero(particles->size()*2,particles->size()*2);

    m_invM.reset(particles->size() * 2, particles->size() * 2);

    for (int i = 0; i < particles->size(); i++) {

        // Down the diagonal
        Particle *p = particles->at(i);
        m_invM_Eigen(2*i,2*i) = contact ? p->tmass : p->imass;
        m_invM_Eigen(2*i+1,2*i+1) = contact ? p->tmass : p->imass;
        m_invM.setValue(2 * i, 2 * i, contact ? p->tmass : p->imass);
        m_invM.setValue(2 * i + 1, 2 * i + 1, contact ? p->tmass : p->imass);
    }
    //std::cout << m_invM_Eigen.format(CleanFmt) << std::endl;
}

void Solver::setupSizes(int numParts, QList<Constraint *> *constraints)
{
    bool change = true;
    int numCons = constraints->size();

    // Only update some things if the number of particles changed
    if (m_nParts != numParts) {
        m_nParts = numParts;
        delete[] m_dp;
        delete[] m_counts;
        m_dp = new double[m_nParts * 2];
        m_dp_Eigen.resize(m_nParts*2);
        m_dp_Eigen.Zero(m_nParts*2);
        m_counts = new int[m_nParts];
        change = true;
    }

    // Update how many constraints affect each particle
    for (int i = 0; i < m_nParts; i++) {
        m_counts[i] = 0;
    }
    for (int i = 0; i < numCons; i++) {
        constraints->at(i)->updateCounts(m_counts);
    }

    if (m_nCons != numCons) {
        m_nCons = numCons;
        delete[] m_b;
        delete[] m_gamma;
        m_b = new double[m_nCons];
        m_b_Eigen.resize(m_nCons);
        m_b_Eigen.Zero(m_nCons);
        m_gamma = new double[m_nCons];
        m_gamma_Eigen.resize(m_nCons);
        m_gamma_Eigen.Zero(m_nCons);
        change = true;
    }

    if (change) {
        m_JT.reset(m_nParts * 2, m_nCons);
        m_JT_Eigen.resize(m_nParts * 2, m_nCons);
        m_JT_Eigen.Zero(m_nParts * 2, m_nCons);
    }
}

void Solver::solveAndUpdate(QList<Particle *> *particles, QList<Constraint *> *constraints, bool stabile)
{
    if (constraints->size() == 0) {
        return;
    }

    //std::cout << m_invM_Eigen.format(CleanFmt) << std::endl;

    // Reset J^T and b
    bool updated = false;

    // Loop!
    for (int i = 0; i < particles->size(); i++) {
        for (int j = 0; j < constraints->size(); j++) {
            Constraint *cons = constraints->at(j);

            // Update b
            if (!updated) {
                //m_b[j] = -cons->evaluate(particles);
                m_b_Eigen(j) = -cons->evaluate(particles);
            }
            glm::vec2 grad_ji = cons->gradient(particles, i);
            //m_JT.setValue(2 * i, j, grad_ji.x);
            m_JT_Eigen(2*i,j) = grad_ji.x;
            m_JT_Eigen(2*i+1,j) = grad_ji.y;
            //m_JT.setValue(2 * i + 1, j, grad_ji.y);
        }
        updated = true;
    }

    //SparseMatrix temp = m_invM * m_JT;
    Eigen::MatrixXd temp_Eigen = m_invM_Eigen * m_JT_Eigen;
    Eigen::MatrixXd m_A_Eigen = m_JT_Eigen.transpose() * temp_Eigen;
    //m_A = m_JT.getTranspose() * temp;
//    m_JT.printMatrix(4,false);
    //m_eq.setA(&m_A);
//    cout << endl;
//    for (int i = 0; i < particles->size(); i++) {
//        printf("%.4f\n", m_b[i]);
//    }
    //bool result = m_eq.solve(m_b, m_gamma);
    m_gamma_Eigen = m_A_Eigen.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(m_b_Eigen);
//    cout << result << endl;
//    for (int i = 0; i < particles->size(); i++) {
//        printf("%.4f\n", m_gamma[i]);
//    }
//    cout << endl;
    //temp.multiply(m_dp, m_gamma, particles->size() * 2, 1);

    m_dp_Eigen = temp_Eigen * m_gamma_Eigen;

    for (int i = 0; i < particles->size(); i++) {
        Particle *p = particles->at(i);
        int n = m_counts[i];
        double mult = n > 0 ? (RELAXATION_PARAMETER / (double)n) : 0.,
               dx_Eigen = m_dp_Eigen(2*i) * mult,
               //dx = m_dp[2 * i] * mult,
               dy_Eigen = m_dp_Eigen(2*i+1) * mult;
               //dy = m_dp[2 * i + 1] * mult;

        if (std::isnan(dx_Eigen)) dx_Eigen = 0.0;//cout << "Nan Found dx_Eigen" << endl;
        if (std::isnan(dy_Eigen)) dy_Eigen = 0.0;//cout << "Nan Found dy_Eigen" << endl;
        //cout << i << " " << dx << " " << dy << " " << dx_Eigen << " " << dy_Eigen << endl;

        //p->ep.x += (fabs(dx) > EPSILON ? dx : 0);
        //p->ep.y += (fabs(dy) > EPSILON ? dy : 0);
        p->ep.x += (fabs(dx_Eigen) > EPSILON ? dx_Eigen : 0);
        p->ep.y += (fabs(dy_Eigen) > EPSILON ? dy_Eigen : 0);
        if (stabile) {
           // p->p.x += (fabs(dx) > EPSILON ? dx : 0);
            //p->p.y += (fabs(dy) > EPSILON ? dy : 0);
        }
    }
}
