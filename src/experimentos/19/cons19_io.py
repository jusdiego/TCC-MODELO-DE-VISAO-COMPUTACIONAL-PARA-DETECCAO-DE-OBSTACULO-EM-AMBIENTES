"""Drive, recibos atômicos e pacote leve; credenciais ficam fora dos resultados."""
from __future__ import annotations
import base64
import configparser
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import diag11_core as core

class DigestMismatch(RuntimeError):
    pass

class Drive:
    def __init__(self, binary, config, root='gdrive_tcc:TCC_Visao_Computacional'):
        self.binary, self.config, self.root = str(binary), str(config), root

    @classmethod
    def from_kaggle(cls):
        from kaggle_secrets import UserSecretsClient
        root = Path('/tmp/tcc_rclone')
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root.chmod(0o700)
        try:
            secret = UserSecretsClient().get_secret('GDRIVE_RCLONE_CONFIG_B64')
        except Exception:
            raise RuntimeError('Habilite o Secret existente GDRIVE_RCLONE_CONFIG_B64 em Add-ons → Secrets.') from None
        try:
            raw = base64.b64decode(secret.strip(), validate=True)
            cfg = configparser.ConfigParser(interpolation=None)
            cfg.read_string(raw.decode('utf-8-sig'))
        except Exception:
            raise RuntimeError('A configuração do Secret rclone não é válida.') from None
        if (not cfg.has_section('gdrive_tcc') or cfg.get('gdrive_tcc', 'type', fallback='') != 'drive'
                or not cfg.get('gdrive_tcc', 'client_id', fallback='')
                or not cfg.get('gdrive_tcc', 'token', fallback='')
                or cfg.get('gdrive_tcc', 'scope', fallback='') != 'drive'):
            raise RuntimeError('O Secret precisa da conexão gdrive_tcc autorizada para leitura e gravação em Meu Drive.')
        cfg.remove_option('gdrive_tcc', 'team_drive')
        config = root / 'gdrive_tcc.conf'
        with config.open('w', encoding='utf-8') as f:
            cfg.write(f)
        config.chmod(0o600)
        del secret, raw, cfg
        binary = root / 'rclone'
        if not binary.is_file():
            executable = shutil.which('rclone')
            if not executable:
                import os
                prefix = [] if os.geteuid() == 0 else ['sudo']
                subprocess.run(prefix + ['apt-get', 'update', '-qq'], check=True)
                subprocess.run(prefix + ['apt-get', 'install', '-y', '--no-install-recommends', 'rclone'], check=True)
                executable = shutil.which('rclone')
            if not executable:
                raise RuntimeError('Executável rclone indisponível após a instalação.')
            shutil.copy2(executable, binary)
            binary.chmod(0o700)
        return cls(binary, config)

    def _call(self, args):
        return subprocess.run([self.binary, '--config', self.config] + list(args)
                              + ['--retries', '2', '--low-level-retries', '3',
                                 '--contimeout', '15s', '--timeout', '60s'],
                              capture_output=True, text=True)

    def exists(self, relative):
        result = self._call(['lsjson', self.root + '/' + relative, '--stat'])
        if result.returncode == 0:
            return True
        if result.returncode in {3, 4}:
            return False
        raise RuntimeError('Falha ao consultar Drive (código ' + str(result.returncode) + '): ' + relative)

    def fetch(self, relative, target, expected_sha=None, optional=False):
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        receipt = target.with_name(target.name + '.download.json')
        if target.is_file() and target.stat().st_size:
            local_sha = core.sha256_file(target)
            prior = {}
            if receipt.is_file():
                try:
                    prior = json.loads(receipt.read_text())
                except (ValueError, OSError):
                    pass
            if ((expected_sha and local_sha == expected_sha)
                    or (not expected_sha and prior.get('origem') == relative and prior.get('sha256') == local_sha)):
                print('Entrada local conferida:', relative, flush=True)
                return target
        if optional and not self.exists(relative):
            return None
        print('Baixando:', relative, flush=True)
        tmp = target.with_name(target.name + '.part')
        result = self._call(['copyto', self.root + '/' + relative, str(tmp)])
        if result.returncode:
            raise RuntimeError('Download interrompido (código ' + str(result.returncode) + '): ' + relative)
        if not tmp.is_file() or not tmp.stat().st_size:
            raise RuntimeError('Download vazio: ' + relative)
        actual = core.sha256_file(tmp)
        if expected_sha and actual != expected_sha:
            tmp.replace(target.with_name(target.name + '.rejeitado_' + actual[:12]))
            raise DigestMismatch('SHA-256 diferente da fonte autenticada: ' + relative)
        tmp.replace(target)
        core.save_json(receipt, {'origem': relative, 'sha256': actual, 'bytes': target.stat().st_size})
        return target

    def backup(self, local, relative, required=True):
        print('Salvando resultados no Drive...', flush=True)
        result = self._call(['copy', str(local), self.root + '/' + relative,
                             '--exclude', '**/*.tmp', '--exclude', '**/*.part', '--exclude', '**/__pycache__/**'])
        if result.returncode and required:
            raise RuntimeError('Falha no salvamento no Drive; resultados locais preservados. Código ' + str(result.returncode))
        return result.returncode == 0

    def restore(self, relative, local):
        if not self.exists(relative):
            return False
        result = self._call(['copy', self.root + '/' + relative, str(local),
                             '--exclude', '*.zip', '--exclude', '**/*.tmp', '--exclude', '**/*.part'])
        if result.returncode:
            raise RuntimeError('Não foi possível recuperar o resultado anterior; código ' + str(result.returncode))
        return True

def package(directory):
    directory=Path(directory);target=directory/'pacote_consolidacao_19.zip';tmp=target.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for path in sorted(directory.rglob('*')):
            if not path.is_file() or path in {target,tmp} or path.suffix in {'.pt','.tmp','.part'}:continue
            rel=path.relative_to(directory)
            if '__pycache__' in rel.parts:continue
            z.write(path,rel.as_posix())
    tmp.replace(target)
    return target

def failure(directory,error,stage,drive=None,remote=None):
    core.save_json(Path(directory)/'execucao_interrompida_19.json',{'concluido':False,'etapa':stage,
        'tipo':type(error).__name__,'mensagem':str(error)[:1400],'treinamento_executado':False,
        'inferencia_executada':False,'reserva_inferida':False,'modelo_substituido':False})
    result=package(directory)
    if drive and remote:
        try:drive.backup(directory,remote,required=False)
        except Exception:pass
    return result
