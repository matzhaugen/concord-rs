use ndarray::linalg::Dot;
use ndarray::Zip;
use ndarray::{Array2, ArrayD, ArrayView2, ArrayViewD, ArrayViewMut2, ArrayViewMutD};
use numpy::{IntoPyArray, PyArray2, PyArrayDyn, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use sprs::binop;
use sprs::CsMat;
use std::time::{Duration, Instant};

#[pymodule]
fn pyconcord(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // immutable example
    fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
        a * &x + &y
    }

    // mutable example (no return)
    fn mult(a: f64, mut x: ArrayViewMutD<'_, f64>) {
        x *= a;
    }

    fn sign(a: &f64) -> f64 {
        if *a > 0.0 {
            return 1.0;
        } else {
            return -1.0;
        }
    }

    fn max_zero(a: f64) -> f64 {
        if a >= 0.0 {
            return a;
        } else {
            return 0.0;
        }
    }

    fn abs(a: f64) -> f64 {
        if a > 0.0 {
            return a;
        } else {
            return -a;
        }
    }

    fn sthreshmat<'a>(mut omega: ArrayViewMut2<f64>, tau: f64, lambda_matrix: ArrayView2<f64>) {
        Zip::from(&mut omega).and(&lambda_matrix).apply(|o, l| {
            if *o > 0.0 {
                *o = max_zero(abs(*o) - tau * l)
            } else {
                *o = -max_zero(abs(*o) - tau * l)
            };
        });
        // omega.mapv_inplace(abs);
        // omega -= &lambda_matrix;
        // omega.mapv_inplace(max_zero);
        // omega *= &sign_omega;
    }

    fn concord(data: ArrayView2<'_, f64>, n: &usize, p: &usize, _lambda: f64) -> Array2<f64> {
        let maxitr = 100;
        // Initial guess - identity matrix
        let mut omega_k = CsMat::eye_csc(*p);
        let mut omega = CsMat::eye_csc(*p);
        let mut s = Array2::<f64>::zeros((*p, *p));
        s = data.t().dot(&data) / (*n as f64);
        let mut start = Instant::now();
        let mut h = s.diag().sum() / 2.0;
        let mut g = &s - &omega_k.to_dense(); // simplified since x is the identity
        let mut g_k;
        let mut w_k = g.clone();
        let mut subg;
        let mut step;
        let mut temp = Array2::<f64>::zeros((*p, *p));
        let mut lambda_matrix = Array2::<f64>::ones([*p, *p]) * _lambda;
        let diagonal = Array2::<f64>::eye(*p) * _lambda;
        lambda_matrix -= &diagonal;
        let mut lambda_matrix_i;
        let mut not_converged = true;
        let mut tau: f64;
        let mut taun = 1.0;
        let mut diagitr: i16 = 0;
        let mut backitr: i16;
        let mut itr: i16 = 0;
        let mut min_elem: f64;
        let mut step_norm: f64;
        let mut q_k: f64;
        let mut h_k: f64 = 0.0;
        let mut log_det: f64;
        let mut trace_product_1: f64;
        let mut trace_product_2: f64;
        let mut subgnorm: f64;
        let mut omega_k_norm: f64;
        let epstol: f64 = 1e-5;
        let zero_threshold: f64 = 1e-6;
        let half = 0.5;
        let mut duration = start.elapsed();
        println!("Time elapsed in expensive_function() is: {:?}", duration);

        while not_converged {
            backitr = 0;
            tau = taun.clone();
            lambda_matrix_i = lambda_matrix.clone();

            while backitr < 10 {
                if diagitr != 0 || backitr != 0 {
                    tau *= half;
                }
                // println!("tau: {:?}", tau);
                start = Instant::now();
                temp = -(&g);
                duration = start.elapsed();
                temp *= tau;

                println!("Time elapsed in expensive_function() is: {:?}", duration);
                temp += &omega.to_dense();

                sthreshmat(temp.view_mut(), tau, lambda_matrix_i.view());

                omega_k = CsMat::csr_from_dense(temp.view(), zero_threshold);

                // println!("omega: {:?}", &omega);
                // println!("omega_k: {:?}", &omega_k.to_dense());
                min_elem = omega_k
                    .diag()
                    .data()
                    .iter()
                    .fold(1.0, |acc, elem| acc.min(*elem));
                if min_elem < 1e-8 && diagitr < 20 {
                    diagitr += 1;
                    continue;
                }

                step = &omega_k - &omega;
                // println!("step: {:.6?}", step.to_dense());

                step_norm = step
                    .iter()
                    .fold(0.0, |acc, (elem, (_idx, _idxptr))| acc + elem * elem)
                    * 0.5
                    / tau;
                // println!("step norm: {:?}", step_norm);
                // println!("g: {:.6?}", &g);
                // println!("step * g: {:?}", &step * &g);
                trace_product_1 = binop::mul_dense_mat_same_ordering(&step, &g, 1.)
                    .iter()
                    .fold(0.0, |acc, a| acc + a);
                // println!("step * g: {:?}", trace_product_1);
                // println!("h: {:?}", h);
                q_k = h + step_norm + trace_product_1;

                log_det = -omega_k
                    .diag()
                    .data()
                    .iter()
                    .fold(0.0, |acc, val| acc + val.log(std::f64::consts::E));
                w_k = omega_k.dot(&s);
                trace_product_2 = binop::mul_dense_mat_same_ordering(&omega_k, &w_k, 1.)
                    .fold(0.0, |acc, a| acc + a);
                h_k = log_det + 0.5 * trace_product_2;

                // println!("h: {:.6}, q: {:.6}", h_k, q_k);
                if h_k > q_k {
                    backitr += 1;
                } else {
                    break;
                }
            }

            g_k = 0.5 * &(&w_k + &w_k.t());
            let mut g_k_diag = g_k.diag_mut();
            for (idx, val) in omega_k.diag().iter() {
                g_k_diag[idx] -= 1.0 / val
            }

            taun = 1.0; // TODO: can be more fancy here

            temp = g_k.clone()
                + binop::mul_dense_mat_same_ordering(&omega_k.map(sign), &lambda_matrix, 1.);

            subg = g_k.clone();
            sthreshmat(subg.view_mut(), 1.0, lambda_matrix.view());
            ndarray::Zip::from(&mut subg)
                .and(&temp)
                .and(&omega_k.to_dense())
                .apply(|a, b, c| {
                    if *c != 0.0 {
                        *a = *b
                    };
                });

            subgnorm = subg.fold(0.0, |acc, a| acc + a * a).sqrt();
            // println!("subnorm: {:.6?}", subgnorm);
            omega_k_norm = omega_k
                .iter()
                .fold(0.0, |acc, (a, (_, _))| acc + a * a)
                .sqrt();
            // println!("omega_k norm: {:.6?}", omega_k_norm);
            not_converged = subgnorm / omega_k_norm > epstol;

            omega = omega_k.clone();
            h = h_k.clone();
            g = g_k.clone();

            itr = itr + 1;
            if itr >= maxitr {
                not_converged = false;
            }
        }
        println!("Converged in {:?} iterations", itr);
        omega.to_dense()
    }

    #[pyfn(m, "concord")]
    fn concord_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        lambda: f64,
    ) -> &'py PyArray2<f64> {
        let shape = x.shape();
        concord(x.as_array(), &shape[0], &shape[1], lambda).into_pyarray(py)
    }

    // wrapper of `axpy`
    #[pyfn(m, "axpy")]
    fn axpy_py<'py>(
        py: Python<'py>,
        a: f64,
        x: PyReadonlyArrayDyn<f64>,
        y: PyReadonlyArrayDyn<f64>,
    ) -> &'py PyArrayDyn<f64> {
        let x = x.as_array();
        let y = y.as_array();
        axpy(a, x, y).into_pyarray(py)
    }

    // wrapper of `mult`
    #[pyfn(m, "mult")]
    fn mult_py(_py: Python<'_>, a: f64, x: &PyArrayDyn<f64>) -> PyResult<()> {
        let x = unsafe { x.as_array_mut() };
        mult(a, x);
        Ok(())
    }

    Ok(())
}
