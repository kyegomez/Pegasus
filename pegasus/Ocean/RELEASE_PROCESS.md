## Release Process

This guide covers how to release ocean to PyPi

#### Increase the version number

1. Create a new PR for the release that upgrades the version in code. In [this file](https://github.com/ocean-core/ocean/blob/main/oceandb/__init__.py) update the ** version **.

```
__version__ = "A.B.C"
```

2. Add the "release" label to this PR
3. Once the PR is merged, tag your commit SHA with the release version

```
git tag A.B.C <SHA>
```

#### Perform the release

1. Push your tag to origin to create the release

```
git push origin A.B.C
```

2. This will trigger a Github action which performs the release
