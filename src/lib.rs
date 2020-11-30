use ndarray::Zip;
use ndarray::{Array, Array1, ArrayD, ArrayViewD, ArrayView1, ArrayView2, Array2, ArrayViewMutD, ArrayViewMut2};
use numpy::{IntoPyArray, PyArrayDyn, PyArray2, PyReadonlyArrayDyn, PyReadonlyArray2};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

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

    fn trace_triple_mult(a: ArrayView2<'_, f64>, 
					    b: ArrayView2<'_, f64>, 
					    c: ArrayView2<'_, f64>, 
					    p: &usize) -> f64 {
		let mut result: f64 = 0.0;
		for i in 0..*p {
			for j in 0..*p {
				for k in 0..*p {
					result += a[[i, j]] * b[[j, k]] * c[[k, i]];
				}
			}
		}
		result / 2.0
	}

	fn sign(a: f64) -> f64 {
		if a > 0.0 {
			return 1.0
		} else {
			return - 1.0
		}
	}

	
	fn max_zero(a: f64) -> f64 {
		if a > 0.0 {
			return a
		} else {
			return 0.0
		}
	}	

	fn abs(a: f64) -> f64 {
		if a > 0.0 {
			return a
		} else {
			return - a
		}
	}

	fn sthreshmat<'a>(sign_omega: ArrayView2<f64>, 
        mut omega: ArrayViewMut2<f64>, 
        lambda_matrix: ArrayView2<f64>) {
		omega.mapv_inplace(abs);
		omega -= &lambda_matrix;
		omega.mapv_inplace(max_zero);
		omega *= &sign_omega;
	}

    fn concord(data: ArrayView2<'_, f64>, n: &usize, _p: &usize, _lambda: f64) -> Array2<f64> {
        
        let maxitr = 3;
        // Initial guess - identity matrix
        let mut omega_k = Array2::<f64>::eye(* _p);
        let mut omega = Array2::<f64>::eye(* _p);

        let s = data.t().dot(&data) / (*n as f64);

        let mut h = trace_triple_mult(omega_k.view(), s.view(), omega_k.view(), _p);
        println!("h: {:?}", h);

        let mut g = s.clone() - omega_k.view(); // simplified since x is the identity
        let mut g_k = g.clone();
        let mut w_k = g.clone();
        let mut subg = g.clone();
        let mut g_k_temp = g.clone();
        let mut diag_temp = Array1::<f64>::ones(* _p).view_mut();
        let mut step = omega_k.clone();
        let mut temp = omega_k.clone();
        let mut lambda_matrix = Array2::<f64>::ones([* _p, * _p]) * _lambda;
        let diagonal = Array2::<f64>::eye(* _p) * _lambda;
        lambda_matrix -= &diagonal;
        let mut lambda_matrix_i = lambda_matrix.clone();
        println!("lambda matrix: {:?}", lambda_matrix_i);
        let mut not_converged = true;
        let mut tau: f64;
        let mut taun = 1.0;
        let mut diagitr: i16 = 0;
        let mut backitr: i16 = 0;
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
        let half = 0.5;
        println!("G: {:?}", g);
        while not_converged {
            println!("itr: {:?}", itr);
            backitr = 0;
            tau = taun.clone();
            lambda_matrix_i = lambda_matrix.clone();
            while backitr < 2 {
                
                if diagitr != 0 || backitr != 0 {
                    tau *= half;
                    lambda_matrix_i *= tau;
                }

                temp = &g * tau;
                temp = &omega - &temp;
                omega_k = temp;
                println!("tmp before: {:?} {:?}", &omega_k[[0,0]], &omega_k[[0,1]]);
                sthreshmat(omega.mapv(sign).view(), omega_k.view_mut(), lambda_matrix_i.view());
                println!("tmp after: {:?} {:?}", &omega_k[[0,0]], &omega_k[[0,1]]);
                
                min_elem = omega_k.diag().fold(0.0, |acc, elem| acc.min(*elem));
                if min_elem < 1e-8 && diagitr < 20 {
                    println!("Found zero or negative diagonal entries, trying again...");
                    diagitr += 1;
                    continue;
                }

                step = &omega_k - &omega;
                
                step_norm = step.fold(0.0, |acc, a| acc + a * a) * 0.5 / tau;
                trace_product_1 = ndarray::Zip::from(&step).and(&g).fold(0.0, |acc, a, b| acc + a * b);
                q_k = h + step_norm + trace_product_1;
                
                log_det = - omega_k.diag().fold(0.0, |acc, b| acc + b.log(std::f64::consts::E));
                w_k = omega_k.dot(&s);
                trace_product_2 = ndarray::Zip::from(&w_k).and(&omega_k).fold(0.0, |acc, a, b| acc + a * b);
                h_k = log_det + trace_product_2;
                
                println!("h: {:?}, q: {:?}", h_k, q_k);
                if h_k > q_k {
                    backitr += 1;
                } else {
                    println!("Backtrace finished with tau {} after {} iterations ", tau, backitr);
                    break;

                }
                
            }

            
            g_k = 0.5 * &(&w_k + &w_k.t());
            diag_temp = g_k.diag_mut();
            diag_temp -= &omega_k.diag().mapv(|a| 1.0 / a);

            taun = 1.0; // TOGO: can be more fancy here

            temp = omega_k.mapv(sign);
            temp *= &lambda_matrix; // First term
            temp += &g_k; 

            subg = g_k.clone();
            sthreshmat(subg.mapv(sign).view(), subg.view_mut(), lambda_matrix.view());
            ndarray::Zip::from(&mut subg).and(& temp).and(& omega_k).apply(|a, b, c| {
                if *c != 0.0 { b } else { a };
            });

            subgnorm = subg.fold(0.0, |acc, a| acc + a * a);
            omega_k_norm = omega_k.fold(0.0, |acc, a| acc + a * a);


            not_converged = subgnorm / omega_k_norm > epstol;
            println!("Covergence Ratio: {:?}", subgnorm / omega_k_norm);
            
            omega = omega_k.clone();
            h = h_k.clone();
            g = g_k.clone();

        	itr = itr + 1;
        	if itr >= maxitr {
        		not_converged = false;
        	}
        };


        omega
    }

    #[pyfn(m, "concord")]
    fn concord_py<'py>(
    	py: Python<'py>,
    	x: PyReadonlyArray2<f64>,
    	lambda: f64
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