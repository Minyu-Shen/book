// 保持侧边栏滚动位置
(function() {
    var sidebar = document.querySelector('.sidebar');
    if (!sidebar) return;

    // 恢复上次保存的滚动位置
    var saved = sessionStorage.getItem('sidebar-scroll');
    if (saved !== null) {
        sidebar.scrollTop = parseInt(saved, 10);
    }

    // 点击侧边栏链接前保存滚动位置
    sidebar.addEventListener('click', function(e) {
        if (e.target.tagName === 'A') {
            sessionStorage.setItem('sidebar-scroll', sidebar.scrollTop);
        }
    });
})();

// 为所有代码块自动添加复制按钮
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('pre > code').forEach(function(codeBlock) {
        var pre = codeBlock.parentElement;
        pre.style.position = 'relative';

        var btn = document.createElement('button');
        btn.className = 'code-copy-btn';
        btn.textContent = '复制';
        btn.addEventListener('click', function() {
            navigator.clipboard.writeText(codeBlock.textContent).then(function() {
                btn.textContent = '已复制';
                btn.classList.add('copied');
                setTimeout(function() {
                    btn.textContent = '复制';
                    btn.classList.remove('copied');
                }, 2000);
            });
        });
        pre.appendChild(btn);
    });
});
