{
  description = "Scheduling machine learning workloads in distributed environments";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";

    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.05";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    ...
  } @ inputs:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = inputs.nixpkgs.legacyPackages.${system};
    in {
      devShells.default = pkgs.mkShell {
        packages = with pkgs; [
          (pkgs.python311.withPackages (python-pkgs: [
            python-pkgs.flask
            python-pkgs.requests
            python-pkgs.waitress
            python-pkgs.psycopg2
            python-pkgs.boto3
            python-pkgs.opencv4
            python-pkgs.transformers
            python-pkgs.pytorch
            python-pkgs.langchain
            python-pkgs.openai
            python-pkgs.pypdf
            python-pkgs.datasets
            python-pkgs.torchmetrics
            python-pkgs.albumentations
            python-pkgs.timm
            python-pkgs.accelerate
          ]))

          flutter
        ];
      };
    });
}


