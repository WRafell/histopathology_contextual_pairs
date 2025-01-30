from omegaconf import OmegaConf
import hydra


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: OmegaConf) -> None:
    return


if __name__=='__main__':
    main()
