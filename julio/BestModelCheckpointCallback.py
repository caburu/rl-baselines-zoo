import os, shutil
from stable_baselines.common.callbacks import BaseCallback

class BestModelCheckPointCallback(BaseCallback):
    """
    Callback que faz uma cópia do melhor modelo encontrado a cada `save_freq` passos.
    :param save_freq: (int) Frequência de cópia do melhor modelo.
    :param best_model_path: (str) Caminho para o melhor modelo (a ser copiado).
    :param vecnormalize_path: (str) Caminho para os dados do VecNormalize do melhor modelo (a ser copiado).
    :param copy_path: (str) Caminho para a pasta onde o melhor modelo será copiado.
    :param name_prefix: (str) Prefixo a ser usado nos modelos salvos.
    """
    def __init__(self, save_freq: int, best_model_path: str, vecnormalize_path: str, copy_path: str, name_prefix='best_model', verbose=0):
        super(BestModelCheckPointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.best_model_path = best_model_path
        self.vecnormalize_path = vecnormalize_path
        self.copy_path = copy_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.copy_path is not None:
            os.makedirs(self.copy_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.copy_path, '{}_{}_steps'.format(self.name_prefix, self.num_timesteps))
            shutil.copy2(os.path.join(self.best_model_path, 'best_model.zip'), path+'.zip')
            shutil.copy2(os.path.join(self.vecnormalize_path, 'vecnormalize.pkl'), path+'_vecnormalize.pkl')
            if self.verbose > 1:
                print("Copiando melhor modelo até agora para {}".format(path))
        return True
