
from gym.envs.registration import register


entry_point = 'atrp:ATRPTargetDistribution'
max_episode_steps = 100000
kwargs_common = {
    'max_rad_len': 100,
    'step_time': 1e2,
    'completion_time': 1e5,
    'min_steps': 100,
    'termination': False,
    'k_prop': 1.6e3,
    'k_act': 0.45,
    'k_deact': 1.1e7,
    'k_ter': 1e8,
    'observation_mode': 'all stable',
    'mono_init': 0.0,
    'cu1_init': 0.0,
    'cu2_init': 0.0,
    'dorm1_init': 0.0,
    'mono_unit': 0.1,
    'cu1_unit': 0.004,
    'cu2_unit': 0.004,
    'dorm1_unit': 0.008,
    'mono_cap': 10.0,
    'cu1_cap': 0.2,
    'cu2_cap': 0.2,
    'dorm1_cap': 0.4,
    'mono_density': 8.73,
    'sol_init': 0.01,
    'sol_cap': 0.0,
    'reward_chain_type': 'dorm',
    'ks_num_sample_loose': 2e3,
    'ks_num_sample_tight': 1e4,
}

var24 = [6.31782376e-11,   1.47405524e-09,   1.72059392e-08,   1.33967606e-07,
         7.82762606e-07,   3.66098015e-06,   1.42767649e-05,   4.77486464e-05,
         1.39812219e-04,   3.64099828e-04,   8.53849063e-04,   1.82133872e-03,
         3.56329890e-03,   6.43859203e-03,   1.08089319e-02,   1.69452487e-02,
         2.49183604e-02,   3.45060788e-02,   4.51523327e-02,   5.60033617e-02,
         6.60237794e-02,   7.41693469e-02,   7.95738755e-02,   8.17024395e-02,
         8.04336843e-02,   7.60557676e-02,   6.91848805e-02,   6.06339917e-02,
         5.12674045e-02,   4.18734875e-02,   3.30768688e-02,   2.52975249e-02,
         1.87521175e-02,   1.34853951e-02,   9.41705721e-03,   6.39114814e-03,
         4.21898033e-03,   2.71103068e-03,   1.69697400e-03,   1.03544921e-03,
         6.16281446e-04,   3.58010527e-04,   2.03112001e-04,   1.12601193e-04,
         6.10311077e-05,   3.23581050e-05,   1.67900252e-05,   8.53021031e-06,
         4.24525053e-06,   2.07047078e-06,   9.90002973e-07,   4.64277673e-07,
         2.13628204e-07,   9.64801872e-08,   4.27828530e-08,   1.86337848e-08,
         7.97396796e-09,   3.35372371e-09,   1.38673029e-09,   5.63892942e-10,
         2.25561502e-10,   8.87802988e-11,   3.43927818e-11,   1.31168384e-11,
         4.92618558e-12,   1.82228936e-12,   6.64126670e-13,   2.38512157e-13,
         8.44291147e-14,   2.94638430e-14,   1.01389426e-14,   3.44104370e-15,
         1.15204030e-15,   3.80547172e-16,   1.24049159e-16,   3.99119237e-17,
         1.26768683e-17,   3.97554593e-18,   1.23120440e-18,   3.76602697e-19,
         1.13795770e-19,   3.39723907e-20,   1.00218904e-20,   2.92187362e-21,
         8.42024352e-22,   2.39883884e-22,   6.75695540e-23,   1.88206064e-23,
         5.18449522e-24,   1.41262252e-24,   3.80756939e-25,   1.01537316e-25,
         2.67924646e-26,   6.99617510e-27,   1.80809202e-27,   4.62532293e-28,
         1.17131604e-28,   2.93673324e-29,   7.29054903e-30,   1.79228618e-30,]
kwargs_var24 = kwargs_common.copy()
kwargs_var24['dn_distribution'] = var24
register(
    id='ATRP-ps-td-var24-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_var24
)

var26 = [1.53326528e-10,   3.34026207e-09,   3.64635447e-08,   2.65983463e-07,
         1.45876659e-06,   6.41724050e-06,   2.35906208e-05,   7.45525501e-05,
         2.06795673e-04,   5.11544442e-04,   1.14274363e-03,   2.32897407e-03,
         4.36709297e-03,   7.58769947e-03,   1.22896498e-02,   1.86525504e-02,
         2.66481929e-02,   3.59786630e-02,   4.60661655e-02,   5.61070579e-02,
         6.51840904e-02,   7.24130643e-02,   7.70895025e-02,   7.88010466e-02,
         7.74815747e-02,   7.33993617e-02,   6.70882915e-02,   5.92431986e-02,
         5.06051545e-02,   4.18600006e-02,   3.35658405e-02,   2.61157060e-02,
         1.97330696e-02,   1.44921531e-02,   1.03526235e-02,   7.19881026e-03,
         4.87595132e-03,   3.21903525e-03,   2.07264437e-03,   1.30229223e-03,
         7.98944839e-04,   4.78827555e-04,   2.80488392e-04,   1.60669849e-04,
         9.00408335e-05,   4.93883250e-05,   2.65262269e-05,   1.39563837e-05,
         7.19597997e-06,   3.63742375e-06,   1.80320983e-06,   8.77007057e-07,
         4.18615390e-07,   1.96168131e-07,   9.02784377e-08,   4.08148857e-08,
         1.81327823e-08,   7.91862644e-09,   3.40015798e-09,   1.43592808e-09,
         5.96578409e-10,   2.43902958e-10,   9.81502492e-11,   3.88863766e-11,
         1.51718712e-11,   5.83065495e-12,   2.20764973e-12,   8.23707510e-13,
         3.02927502e-13,   1.09829117e-13,   3.92642466e-14,   1.38440641e-14,
         4.81503924e-15,   1.65229333e-15,   5.59505930e-16,   1.86994834e-16,
         6.16931763e-17,   2.00955849e-17,   6.46385773e-18,   2.05343259e-18,
         6.44369376e-19,   1.99766096e-19,   6.11936348e-20,   1.85247327e-20,
         5.54269656e-21,   1.63936083e-21,   4.79371056e-22,   1.38602509e-22,
         3.96304848e-23,   1.12073378e-23,   3.13505336e-24,   8.67581535e-25,
         2.37547768e-25,   6.43603276e-26,   1.72568891e-26,   4.57965836e-27,
         1.20303526e-27,   3.12856938e-28,   8.05530432e-29,   2.05367886e-29,]
kwargs_var26 = kwargs_common.copy()
kwargs_var26['dn_distribution'] = var26
register(
    id='ATRP-ps-td-var26-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_var26
)

var28 = [2.95686388e-10,   6.10952723e-09,   6.33231073e-08,   4.39093450e-07,
         2.29231053e-06,   9.61334379e-06,   3.37463183e-05,   1.02024166e-04,
         2.71267870e-04,   6.44596836e-04,   1.38643278e-03,   2.72721074e-03,
         4.94840623e-03,   8.34192360e-03,   1.31457521e-02,   1.94677446e-02,
         2.72173707e-02,   3.60674414e-02,   4.54622920e-02,   5.46770374e-02,
         6.29182356e-02,   6.94445049e-02,   7.36802558e-02,   7.52981363e-02,
         7.42547485e-02,   7.07763729e-02,   6.53029698e-02,   5.84065449e-02,
         5.07026324e-02,   4.27714881e-02,   3.51001870e-02,   2.80502707e-02,
         2.18497248e-02,   1.66040578e-02,   1.23194365e-02,   8.93093907e-03,
         6.33035550e-03,   4.38989156e-03,   2.98003019e-03,   1.98131310e-03,
         1.29078854e-03,   8.24350446e-04,   5.16287687e-04,   3.17210827e-04,
         1.91258408e-04,   1.13197756e-04,   6.57837258e-05,   3.75467927e-05,
         2.10525380e-05,   1.15987514e-05,   6.28039008e-06,   3.34287763e-06,
         1.74944352e-06,   9.00346998e-07,   4.55757459e-07,   2.26962227e-07,
         1.11211686e-07,   5.36296111e-08,   2.54563407e-08,   1.18960721e-08,
         5.47401723e-09,   2.48074361e-09,   1.10740995e-09,   4.87038067e-10,
         2.11067745e-10,   9.01490594e-11,   3.79538723e-11,   1.57536426e-11,
         6.44776993e-12,   2.60264246e-12,   1.03625781e-12,   4.07043118e-13,
         1.57761962e-13,   6.03423833e-14,   2.27807951e-14,   8.49001569e-15,
         3.12397242e-15,   1.13508870e-15,   4.07324073e-16,   1.44377899e-16,
         5.05560739e-17,   1.74911559e-17,   5.97992495e-18,   2.02052146e-18,
         6.74806307e-19,   2.22791763e-19,   7.27241824e-20,   2.34732205e-20,
         7.49262571e-21,   2.36545875e-21,   7.38700188e-22,   2.28214434e-22,
         6.97573141e-23,   2.10987591e-23,   6.31527853e-24,   1.87086992e-24,
         5.48600731e-25,   1.59248842e-25,   4.57665208e-26,   1.30231203e-26,]
kwargs_var28 = kwargs_common.copy()
kwargs_var28['dn_distribution'] = var28
register(
    id='ATRP-ps-td-var28-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_var28
)

var31 = [3.14319718e-10,   6.58362722e-09,   6.90967661e-08,   4.84538191e-07,
         2.55430596e-06,   1.07986741e-05,   3.81421069e-05,   1.15790457e-04,
         3.08461521e-04,   7.32673919e-04,   1.57141662e-03,   3.07479700e-03,
         5.53625025e-03,   9.23974731e-03,   1.43844288e-02,   2.10050265e-02,
         2.89133820e-02,   3.76839033e-02,   4.66937343e-02,   5.52112671e-02,
         6.25114008e-02,   6.79886028e-02,   7.12415436e-02,   7.21136753e-02,
         7.06876988e-02,   6.72433239e-02,   6.21939226e-02,   5.60181393e-02,
         4.91988656e-02,   4.21766469e-02,   3.53196499e-02,   2.89089117e-02,
         2.31358751e-02,   1.81087803e-02,   1.38647378e-02,   1.03848144e-02,
         7.60998625e-03,   5.45629986e-03,   3.82804949e-03,   2.62824858e-03,
         1.76612216e-03,   1.16173817e-03,   7.48184165e-04,   4.71858480e-04,
         2.91485978e-04,   1.76414285e-04,   1.04634248e-04,   6.08354483e-05,
         3.46820323e-05,   1.93928615e-05,   1.06388492e-05,   5.72781680e-06,
         3.02727121e-06,   1.57110954e-06,   8.00900161e-07,   4.01133773e-07,
         1.97450932e-07,   9.55446221e-08,   4.54617390e-08,   2.12760875e-08,
         9.79611980e-09,   4.43855002e-09,   1.97951325e-09,   8.69178483e-10,
         3.75831563e-10,   1.60069740e-10,   6.71664614e-11,   2.77725705e-11,
         1.13185521e-11,   4.54742344e-12,   1.80146556e-12,   7.03812148e-13,
         2.71231345e-13,   1.03122968e-13,   3.86884617e-14,   1.43249887e-14,
         5.23562204e-15,   1.88919198e-15,   6.73113999e-16,   2.36850955e-16,
         8.23196093e-17,   2.82643636e-17,   9.58842944e-18,   3.21433583e-18,
         1.06495750e-18,   3.48763430e-19,   1.12913615e-19,   3.61439981e-20,
         1.14408194e-20,   3.58149619e-21,   1.10895133e-21,   3.39667653e-22,
         1.02929911e-22,   3.08621123e-23,   9.15706400e-24,   2.68894753e-24,
         7.81541135e-25,   2.24859796e-25,   6.40484108e-26,   1.80628683e-26,]
kwargs_var31 = kwargs_common.copy()
kwargs_var31['dn_distribution'] = var31
register(
    id='ATRP-ps-td-var31-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_var31
)

var36 = [6.88942786e-10,   1.37581929e-08,   1.37644476e-07,   9.20015483e-07,
         4.62289671e-06,   1.86314791e-05,   6.27536231e-05,   1.81740954e-04,
         4.62157191e-04,   1.04870141e-03,   2.15087923e-03,   4.02948156e-03,
         6.95631437e-03,   1.11501080e-02,   1.67030650e-02,   2.35204496e-02,
         3.12952892e-02,   3.95307792e-02,   4.76079962e-02,   5.48822598e-02,
         6.07835455e-02,   6.48971205e-02,   6.70086435e-02,   6.71093722e-02,
         6.53675316e-02,   6.20783460e-02,   5.76068603e-02,   5.23353750e-02,
         4.66228959e-02,   4.07792918e-02,   3.50531386e-02,   2.96300713e-02,
         2.46378148e-02,   2.01544936e-02,   1.62177978e-02,   1.28336385e-02,
         9.98376416e-03,   7.63230941e-03,   5.73144814e-03,   4.22630694e-03,
         3.05920208e-03,   2.17317026e-03,   1.51472504e-03,   1.03579166e-03,
         6.94832855e-04,   4.57248992e-04,   2.95193234e-04,   1.86972056e-04,
         1.16202349e-04,   7.08737408e-05,   4.24290270e-05,   2.49362753e-05,
         1.43906716e-05,   8.15655385e-06,   4.54158445e-06,   2.48476052e-06,
         1.33610674e-06,   7.06286778e-07,   3.67120246e-07,   1.87683736e-07,
         9.43928217e-08,   4.67140203e-08,   2.27537030e-08,   1.09106995e-08,
         5.15164851e-09,   2.39567919e-09,   1.09747276e-09,   4.95374253e-10,
         2.20362521e-10,   9.66261629e-11,   4.17725110e-11,   1.78077682e-11,
         7.48744734e-12,   3.10558825e-12,   1.27092042e-12,   5.13255964e-13,
         2.04581243e-13,   8.04983628e-14,   3.12729816e-14,   1.19972711e-14,
         4.54563878e-15,   1.70127066e-15,   6.29047151e-16,   2.29820132e-16,
         8.29755955e-17,   2.96095013e-17,   1.04445524e-17,   3.64237719e-18,
         1.25595300e-18,   4.28263495e-19,   1.44428383e-19,   4.81782841e-20,
         1.58986310e-20,   5.19073671e-21,   1.67690837e-21,   5.36105036e-22,
         1.69628503e-22,   5.31255397e-23,   1.64706116e-23,   5.05549033e-24,]
kwargs_var36 = kwargs_common.copy()
kwargs_var36['dn_distribution'] = var36
register(
    id='ATRP-ps-td-var36-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_var36
)

var41 = [4.48698736e-09,   7.82447792e-08,   6.85361659e-07,   4.02190348e-06,
         1.77952203e-05,   6.33493433e-05,   1.89093818e-04,   4.87046806e-04,
         1.10569561e-03,   2.24908624e-03,   4.15348710e-03,   7.04045918e-03,
         1.10560989e-02,   1.62149650e-02,   2.23685004e-02,   2.92102327e-02,
         3.63171565e-02,   4.32146381e-02,   4.94461689e-02,   5.46309317e-02,
         5.84992566e-02,   6.09045248e-02,   6.18163477e-02,   6.13023670e-02,
         5.95053230e-02,   5.66197796e-02,   5.28706538e-02,   4.84943308e-02,
         4.37226646e-02,   3.87701611e-02,   3.38246506e-02,   2.90415771e-02,
         2.45416830e-02,   2.04115165e-02,   1.67059838e-02,   1.34521653e-02,
         1.06537537e-02,   8.29567941e-03,   6.34866105e-03,   4.77352694e-03,
         3.52519119e-03,   2.55617275e-03,   1.81954700e-03,   1.27124235e-03,
         8.71640198e-04,   5.86497572e-04,   3.87270224e-04,   2.50957434e-04,
         1.59610337e-04,   9.96435229e-05,   6.10701662e-05,   3.67516137e-05,
         2.17207454e-05,   1.26099785e-05,   7.19267767e-06,   4.03183252e-06,
         2.22152673e-06,   1.20348564e-06,   6.41171970e-07,   3.36013322e-07,
         1.73256909e-07,   8.79183828e-08,   4.39162995e-08,   2.15987782e-08,
         1.04613790e-08,   4.99116779e-09,   2.34619570e-09,   1.08684736e-09,
         4.96256952e-10,   2.23392122e-10,   9.91607447e-11,   4.34118070e-11,
         1.87480284e-11,   7.98848724e-12,   3.35903331e-12,   1.39406375e-12,
         5.71143715e-13,   2.31034429e-13,   9.22887721e-14,   3.64109923e-14,
         1.41904642e-14,   5.46397205e-15,   2.07890141e-15,   7.81694623e-16,
         2.90523282e-16,   1.06739876e-16,   3.87734403e-17,   1.39271577e-17,
         4.94730265e-18,   1.73823444e-18,   6.04139096e-19,   2.07734153e-19,
         7.06762893e-20,   2.37950074e-20,   7.92856395e-21,   2.61486383e-21,
         8.53685169e-22,   2.75922928e-22,   8.83009879e-23,   2.79818522e-23,]
kwargs_var41 = kwargs_common.copy()
kwargs_var41['dn_distribution'] = var41
register(
    id='ATRP-ps-td-var41-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_var41
)

var49 = [4.27536053e-09,   7.77258204e-08,   7.07354785e-07,   4.29710393e-06,
         1.96061979e-05,   7.16791785e-05,   2.18774431e-04,   5.73528863e-04,
         1.31877652e-03,   2.70314739e-03,   5.00358248e-03,   8.45409966e-03,
         1.31585405e-02,   1.90194988e-02,   2.57160462e-02,   3.27464174e-02,
         3.95257673e-02,   4.55066954e-02,   5.02825516e-02,   5.36432582e-02,
         5.55741421e-02,   5.62093233e-02,   5.57636812e-02,   5.44682304e-02,
         5.25255911e-02,   5.00908060e-02,   4.72732726e-02,   4.41507601e-02,
         4.07863363e-02,   3.72418123e-02,   3.35849217e-02,   2.98903844e-02,
         2.62366412e-02,   2.27004732e-02,   1.93513867e-02,   1.62469987e-02,
         1.34300303e-02,   1.09270389e-02,   8.74872377e-03,   6.89148447e-03,
         5.33984777e-03,   4.06937463e-03,   3.04969310e-03,   2.24736323e-03,
         1.62835760e-03,   1.16002575e-03,   8.12492583e-04,   5.59509805e-04,
         3.78830007e-04,   2.52202729e-04,   1.65101897e-04,   1.06288526e-04,
         6.72967705e-05,   4.19104882e-05,   2.56758016e-05,   1.54758542e-05,
         9.17856825e-06,   5.35733259e-06,   3.07781208e-06,   1.74070120e-06,
         9.69316792e-07,   5.31543614e-07,   2.87089433e-07,   1.52747808e-07,
         8.00731982e-08,   4.13646188e-08,   2.10608503e-08,   1.05706832e-08,
         5.23100898e-09,   2.55268764e-09,   1.22860891e-09,   5.83319787e-10,
         2.73243180e-10,   1.26302856e-10,   5.76194279e-11,   2.59469528e-11,
         1.15354387e-11,   5.06383816e-12,   2.19528031e-12,   9.40004657e-13,
         3.97616274e-13,   1.66171120e-13,   6.86224430e-14,   2.80063738e-14,
         1.12976307e-14,   4.50522248e-15,   1.77623716e-15,   6.92465077e-16,
         2.66969897e-16,   1.01800152e-16,   3.83980731e-17,   1.43283916e-17,
         5.29009634e-18,   1.93267118e-18,   6.98762790e-19,   2.50050185e-19,
         8.85722096e-20,   3.10589686e-20,   1.07830424e-20,   3.70684935e-21,]
kwargs_var49 = kwargs_common.copy()
kwargs_var49['dn_distribution'] = var49
register(
    id='ATRP-ps-td-var49-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_var49
)
