# Tolmacz model

Offline Polish-English translation model for the [Tolmacz .NET library](https://github.com/TomJerzak/tolmacz).
One model translates in both directions: Polish to English and English to Polish.

## Available model

| Release | Languages | Download size | Extracted size | License |
| --- | --- | ---: | ---: | --- |
| [`bidi-eng-pol-v1`](https://github.com/TomJerzak/tolmacz-model/releases/tag/bidi-eng-pol-v1) | Polish to English, English to Polish | 320 MB | 483 MB | CC BY 4.0 |

## Install

1. Download `bidi-eng-pol-v1.zip` and `bidi-eng-pol-v1.zip.sha256` from the release.
2. Check the download:

   ```bash
   shasum -a 256 -c bidi-eng-pol-v1.zip.sha256
   ```

   On Windows PowerShell, compare the result with the value in the `.sha256` file:

   ```powershell
   Get-FileHash .\bidi-eng-pol-v1.zip -Algorithm SHA256
   ```

3. Extract all files into one folder, for example `models/bidi-eng-pol`.

## Use with Tolmacz

Add the [Tolmacz NuGet package](https://www.nuget.org/packages/Tolmacz) to your project with
`dotnet add package Tolmacz`, then pass it the model folder. The library's source code is in the
[Tolmacz repository](https://github.com/TomJerzak/tolmacz). If you only want to translate, download
the ready-to-use app with the model included from the
[Tolmacz releases](https://github.com/TomJerzak/tolmacz/releases/latest).

```csharp
using Tolmacz;

const string modelDirectory = "models/bidi-eng-pol";

await TranslationModel.VerifyChecksumsAsync(modelDirectory);

using var translator = new MarianOnnxTranslator(modelDirectory);
var result = await translator.TranslateAsync(
    "Dzień dobry.",
    TranslationDirection.PolishToEnglish);

Console.WriteLine(result.Text);
```

The library runs the model locally and does not download files by itself.

## How the model was made

`scripts/prepare_model.py` downloads the original model, converts it to ONNX, quantizes its
weights to 8-bit integers, and writes `model.json`.

## License and attribution

The model is based on [BiDi-eng-pol](https://huggingface.co/allegro/BiDi-eng-pol) and is licensed
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See [NOTICE.md](NOTICE.md).

Machine translation can contain errors. The original authors warn that English-to-Polish output
may show gender bias, so review important translations.
