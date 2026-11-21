# Flash-Next auf zwei MI50

Der Profilgenerator erzeugt eine separate Strata-Konfiguration für zwei
Instinct MI50 mit je 32 GiB und einen nativen gfx906-Build. Er übernimmt
die Geräteauswahl, die Modellpfade, den Tokenizer und eine vorhandene
Bildkonfiguration aus der Basisdatei. Die neue Konfiguration bindet auf
Loopback. Die beiden Ein-Slot-Vergleichsprofile sind ebenfalls verfügbar.
[Build für gfx906](AMD_HIP.md#gfx906-instinct-mi50--mi60-radeon-vii-wave64-built-from-source)

## Drei Slots mit IQ3_S

Getestet mit Strata 0.2.17, SC117 Qwen3.8-Flash-Next IQ3_S, einem
passenden MTP-Runtime-Verzeichnis und 128 GiB System-RAM. In einer
vorbereiteten HIP-Konfiguration müssen beide Geräte, der native Pack,
das GGUF-Modell und dessen PLE-Shard ausgewählt sein.

```sh
python3 terminal/tools/dual_mi50_profile.py basis.json mi50-drei.json --three-slot --mtp-dir /pfad/zur/mtp-runtime
python3 terminal/serve/server.py --config mi50-drei.json --host 127.0.0.1 --port 18750
```

Enthält die Basis bereits das passende `--mtp`, kann `--mtp-dir` entfallen.
Beim Start die drei Slots in `/props` prüfen. Im Engine-Log sollten der
geladene MTP-Drafter, der FP16-Promptpfad auf beiden GPUs und zwei
Pipeline-Fenster für den Solo-Pfad erkennbar sein.

| Einstellung | Drei-Slot-Profil |
| --- | --- |
| Schichten und Kontext | Split bei 24, 102.400 Token je Slot, int8-KV |
| Prompt-Verarbeitung | Blockgröße 2.048, FP16-Promptmatrizen, PLE-Tabelle im RAM |
| Gleichzeitige Anfragen | Drei Slots in einer Batch-Gruppe |
| Einzelne laufende Anfrage | MTP mit `--spec 4` und zwei überlappenden Decode-Fenstern |
| Geparkte Dialoge | Bis zu acht, gemeinsam 4.096 MiB, ab 4.096 Kontexttoken |

Die Fenster überlappen die Arbeit der zwei GPUs bei einer einzeln
laufenden Anfrage. Bei mehreren aktiven Anfragen übernimmt der
Batch-Pfad. In einer Gruppe verarbeitet er aktive Slots gemeinsam und
führt das Fenster nacheinander durch beide GPU-Stufen. Bei drei Gruppen
besteht jede Gruppe aus einem Slot. Die Engine kann dann verschiedene
Gruppen gleichzeitig durch die Stufen schicken. Das kann lange Anfragen
beschleunigen und die Latenz einer parallel eintreffenden kurzen Anfrage
erhöhen. Für den Vergleich in einer Kopie der erzeugten Konfiguration
`--batch-groups` von `1` auf `3` ändern und die im Engine-Log gemeldete
Gruppierung prüfen. Die Überlappung des Promptlesens lässt sich in einem
eigenen Durchlauf mit `STRATA_PIPED_CONCURRENT=1` und
`STRATA_PIPED_CONCURRENT_MIN=4096` prüfen.

Für einen eigenen Vergleich zuerst einen neuen langen Prompt einlesen,
danach feste Text- und Codeausgaben messen. Anschließend bei laufender
langer Anfrage eine kurze strukturierte Antwort anfordern und den Versuch
mit einer dritten Anfrage wiederholen. Gesamtzeit und Korrektheit je
Anfrage aufzeichnen. Für den Cache sieben unterschiedliche längere Dialoge
öffnen und den ersten mit einer kurzen Folgefrage fortsetzen. Die
übernommenen Token zeigen, ob der Zustand wiederverwendet wurde.

## Ein-Slot-Vergleich mit Unsloth Q4

Für die nativen Varianten UD-IQ4_XS und UD-Q4_K_XL erzeugt das Werkzeug
ein serielles Profil oder ein Vergleichsprofil mit zwei Decode-Fenstern.
Kontext und KV-Typ stammen hier aus der Basis. In beiden Fällen verwendet
der Drafter die Einstellung `--spec 3`. [Modellvorbereitung](UNSLOTH_Q4.md)

```sh
python3 terminal/tools/dual_mi50_profile.py basis-q4.json mi50-seriell.json
python3 terminal/tools/dual_mi50_profile.py basis-q4.json mi50-pipeline.json --pipeline
```

Die Server einzeln starten und für beide dieselben Eingaben verwenden.
