$root = "c:\Users\win11\Downloads\MetaHackathonAgent\hiro-social-governance\MiroFish"
$files = Get-ChildItem -Path $root -Recurse -Include *.py,*.json,*.html,*.vue,*.js,*.yml -Exclude package-lock.json | Where-Object { $_.FullName -notmatch 'node_modules|__pycache__|\\\.git\\|dist\\' }

foreach ($f in $files) {
    $content = Get-Content $f.FullName -Raw -Encoding UTF8
    if ($content -match 'MiroFish') {
        $newContent = $content -replace 'MiroFish', 'Hiro'
        Set-Content $f.FullName -Value $newContent -Encoding UTF8 -NoNewline
        Write-Host "Updated MiroFish -> Hiro: $($f.Name)"
    }
}

# Also update .env and .env.example
$envFiles = @("$root\.env", "$root\.env.example")
foreach ($ef in $envFiles) {
    if (Test-Path $ef) {
        $content = Get-Content $ef -Raw -Encoding UTF8
        if ($content -match 'MiroFish') {
            $newContent = $content -replace 'MiroFish', 'Hiro'
            Set-Content $ef -Value $newContent -Encoding UTF8 -NoNewline
            Write-Host "Updated MiroFish -> Hiro: $ef"
        }
    }
}

Write-Host "Done replacing MiroFish -> Hiro"
